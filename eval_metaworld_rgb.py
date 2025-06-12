# eval_metaworld_rgb.py  (fixed version)
import os, sys, time, argparse, random, cv2, numpy as np, torch

# Try to force a software GL context for robustness
# os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
# os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
# print("[INFO] Forcing software rendering with GALLIUM_DRIVER=llvmpipe and MESA_GL_VERSION_OVERRIDE=3.3")

# Set environment variables for rendering - GUI vs headless
import argparse
parser_temp = argparse.ArgumentParser(add_help=False) # Use add_help=False to avoid conflict
parser_temp.add_argument("--gui", action="store_true")
parser_temp.add_argument("--save_video", action="store_true")
temp_args, _ = parser_temp.parse_known_args()

# Decide on the renderer based on flags
# Use hardware renderer ONLY if --gui is specified AND --save_video is NOT.
if temp_args.gui and not temp_args.save_video:
    os.environ['MUJOCO_GL'] = 'glfw'
    print("[INFO] Attempting GUI mode with hardware rendering (glfw).")
    print("[INFO] If this hangs, your environment may lack graphics drivers.")
    print("[INFO] Try running with --save_video to generate an MP4 file instead.")
else:
    # Use software renderer for headless collection or video saving
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    if temp_args.save_video:
        print("[INFO] Using headless OSMesa rendering for video saving.")
    else:
        print("[INFO] Using headless OSMesa rendering (no GUI or video).")

# Get the absolute path to the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'TinyVLA')) # Use absolute path
print(f"[INFO] Adding {os.path.join(SCRIPT_DIR, 'TinyVLA')} to python path")

from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from unified_tinyvla import UnifiedTinyVLAModel

# ----------  MetaWorld imports ----------
import metaworld
from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS
import logging # to suppress Model warnings

# --- Set up mujoco rendering ---
try:
    import mujoco_py
    # Set device ID for rendering (optional)
    try:
        mujoco_py.mjviewer.mj_get_device_id = lambda: 0
    except Exception:
        pass  # Ignore if this fails
except ImportError:
    print("Warning: mujoco_py not available")


# Ensure the MetaWorld environment is set up correctly using the v1 API
def setup_env(task_name: str):
    try:
        env_cls = ALL_ENVIRONMENTS[task_name]
        env = env_cls(intervention_id='000', experiment_id='_exp0_seed=0', apply_mod=False)
        return env
    except Exception as e:
        raise RuntimeError(f"Could not create env {task_name}: {e}")


class MetaWorldRGBEvaluator:
    def __init__(self, model_path, checkpoint_path, device="cuda",
                 image_size=336, show_gui=False):
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.show_gui   = show_gui

        print(f"[✓] Device: {self.device}")

        # ---- build model ----
        self.model = UnifiedTinyVLAModel(model_path, mode="action").to(self.device)
        self._load_head(checkpoint_path)
        self.model.eval()

        # ---- helpers ----
        self.tokenizer       = AutoTokenizer.from_pretrained(model_path)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)

        # dummy normalisation (override if you have stats)
        self.action_mean = torch.zeros(4, device=self.device)
        self.action_std  = torch.ones(4,  device=self.device)

    # -------------------------------------------------------------
    def _load_head(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"[!] Checkpoint {ckpt_path} not found → using un-trained head")
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        self.model.base_model.embed_out.load_state_dict(ckpt, strict=False)
        # compile AFTER the weights are loaded
        self.model.base_model.embed_out = torch.compile(self.model.base_model.embed_out)
        print(f"[✓] Diffusion head loaded from {ckpt_path}")

    # -------------------------------------------------------------
    def _preprocess_img(self, rgb):
        pil = Image.fromarray(rgb.astype(np.uint8))
        return self.image_processor(pil, return_tensors="pt")["pixel_values"] \
                  .to(self.device, dtype=torch.float32)

    # -------------------------------------------------------------
    @torch.no_grad()
    def predict_action(self, rgb, prompt, robot_state=None):
        img_t = self._preprocess_img(rgb)
        tok   = self.tokenizer(prompt, return_tensors="pt",
                               padding=True, truncation=True).to(self.device)
        states = torch.zeros((1,7), device=self.device) \
                 if robot_state is None else \
                 torch.tensor(robot_state, device=self.device).unsqueeze(0)
        
        try:
            # For droid_diffusion head, we need to set eval=True for inference
            # Call the base model directly instead of the wrapper
            out = self.model.base_model(input_ids=tok.input_ids,
                                       attention_mask=tok.attention_mask,
                                       images=img_t,
                                       states=states,
                                       eval=True)  # This is key for diffusion head inference

            # Handle different return types
            if isinstance(out, torch.Tensor):
                # Direct tensor return (eval=True mode for diffusion head)
                act_seq = out
            elif isinstance(out, dict):
                # ---- robust tensor extraction ----
                act_seq = out.get("actions", None)
            else:
                act_seq = None

            if act_seq is None:
                print("[!] No action predicted → zero vector")
                return np.zeros(4, dtype=np.float32)

            # Handle different possible shapes
            if len(act_seq.shape) >= 3:  # [batch, seq_len, action_dim]
                act = act_seq[0, 0].cpu()  # first step of first batch
            elif len(act_seq.shape) == 2:  # [batch, action_dim]
                act = act_seq[0].cpu()  # first batch
            else:  # [action_dim]
                act = act_seq.cpu()
                
            # Apply normalization (ensure action_mean and action_std are on CPU or same device)
            act_normalized = act * self.action_std.cpu() + self.action_mean.cpu()
            act_final = act_normalized.numpy()
            return act_final
            
        except Exception as e:
            print(f"[ERROR] Model prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(4, dtype=np.float32)

    # -------------------------------------------------------------
    def collect_trajectory(self, env, prompt, max_steps=150):
        """Runs a simulation episode without rendering to collect trajectory data."""
        obs, total_reward, frames, sim_states = env.reset(), 0.0, [], []
        
        print(f"[Collector] Starting trajectory collection for: {prompt}")
        sim_states.append(env.sim.get_state()) # Save initial state

        for t in range(max_steps):
            # 1. Get image for the policy
            try:
                rgb = env.render(mode='rgb_array', width=self.image_size, height=self.image_size)
            except Exception as e:
                print(f"[!] RGB render failed during collection: {e}, using dummy image")
                rgb = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

            # 2. Predict action (the slow part)
            state = obs[:7] if len(obs) >= 7 else None
            act   = self.predict_action(rgb, prompt, state)

            # 3. Step the environment
            act_padded = np.pad(act, (0, env.action_space.shape[0] - len(act)))
            obs, r, done, info = env.step(act_padded)
            total_reward += r

            # 4. Store the state and the image used for prediction
            sim_states.append(env.sim.get_state())
            frames.append(rgb)
            
            # 5. Print progress occasionally
            if t > 0 and t % 10 == 0:
                print(f"  Collector Step {t}: reward={r:.3f}, total={total_reward:.3f}")

            if done or info.get("success", False):
                if info.get("success", False):
                    print(f"[SUCCESS] Task completed during collection at step {t}!")
                break
                
        return dict(reward=total_reward, steps=t+1,
                    success=info.get("success", False), frames=frames, sim_states=sim_states)

    # -------------------------------------------------------------
    def render_trajectory(self, env, sim_states, fps=30):
        """Replays a trajectory of simulation states for smooth visualization."""
        if not self.show_gui:
            print("[Replay] GUI not enabled, skipping rendering.")
            return

        print(f"\n[Replay] Rendering {len(sim_states)} states. Press Ctrl+C to skip.")
        
        # Set up the environment for replay
        env.reset()
        
        try:
            for i, sim_state in enumerate(sim_states):
                env.sim.set_state(sim_state)
                #################################################################
                env.render(mode='human')
                #################################################################
                print(f"  Replaying frame {i+1}/{len(sim_states)}", end='\r')
                time.sleep(1/fps)
            print("\n[Replay] Finished.")
        except KeyboardInterrupt:
            print("\n[Replay] Skipped by user.")
        except Exception as e:
            print(f"\n[Replay] GUI rendering failed: {e}")
            print("[!] It seems your environment cannot create a GUI window.")
            print("[!] Please try running again with the --save_video flag to create an MP4 file.")

    # -------------------------------------------------------------
    def save_video(self, frames, filename, fps=10):
        """Save a list of RGB frames as an MP4 video."""
        if not frames:
            print(f"[!] No frames to save for {filename}")
            return
            
        print(f"[INFO] Saving video {filename} with {len(frames)} frames...")
        height, width, _ = frames[0].shape
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"[✓] Video saved: {filename}")

# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="lesjie/Llava-Pythia-400M",
        help="path to huggingface model and tokenizer",
    )
    parser.add_argument("--checkpoint_path", default="checkpoints/diff_head_ft.pth")
    parser.add_argument("--task",            default="pick-place-v1")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--prompt",
                    default="Pick up the object and place it at the target.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()

    # Suppress transformers warnings
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    evaluator = MetaWorldRGBEvaluator(args.model_path,
                                      args.checkpoint_path,
                                      show_gui=args.gui)

    env = setup_env(args.task)
    if env is None: return

    for ep in range(args.episodes):
        print(f"\n===== Running Episode {ep+1}/{args.episodes} =====")
        # 1. Collect trajectory data without rendering
        res = evaluator.collect_trajectory(env, args.prompt,
                                           args.max_steps)
        print(f"[Result] success={res['success']} "
              f"reward={res['reward']:.3f} steps={res['steps']}")

        # 2. Render the collected trajectory in the GUI if enabled
        if args.gui and not args.save_video:
            evaluator.render_trajectory(env, res['sim_states'])

        # 3. Save a video from the collected frames if enabled
        if args.save_video:
            out = f"episode_{ep+1}_{args.task}.mp4"
            # Since rendering can fail, let's regenerate frames from the good sim_states
            # This ensures the video isn't black
            print("[Video] Regenerating frames from trajectory for high-quality video...")
            video_frames = []
            # Reset env once before regenerating frames
            env.reset()
            for sim_state in res['sim_states']:
                env.sim.set_state(sim_state)
                try:
                    # Use a higher resolution for the video
                    frame = env.render(mode='rgb_array', width=480, height=480)
                    video_frames.append(frame)
                except Exception as e:
                    print(f"[!] Frame generation failed during video save: {e}")
                    # Append a black frame if rendering still fails for some reason
                    video_frames.append(np.zeros((480, 480, 3), dtype=np.uint8))
            evaluator.save_video(video_frames, out)

if __name__ == "__main__":
    main()
