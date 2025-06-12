# eval_metaworld_rgb.py (Modern V3 API Version)
import os, sys, time, argparse, random, cv2, numpy as np, torch, mujoco, gymnasium as gym

# Set rendering mode
if os.environ.get('DISPLAY') is None:
    os.environ['MUJOCO_GL'] = 'osmesa'
    print("[INFO] No display found. Using OSMesa for headless rendering.")
else:
    os.environ['MUJOCO_GL'] = 'glfw'
    print("[INFO] Display found. Using GLFW for rendering.")

# Get the absolute path to the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'TinyVLA'))
print(f"[INFO] Adding {os.path.join(SCRIPT_DIR, 'TinyVLA')} to python path")

from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from unified_tinyvla import UnifiedTinyVLAModel

# ----------  MetaWorld imports (Modern V3 API) ----------
import metaworld
from metaworld.env_dict import ALL_V3_ENVIRONMENTS
import logging

# Modern MetaWorld v3 API setup
def setup_env(task_name: str, render_mode: str = 'rgb_array'):
    """Sets up a MetaWorld V3 environment using the ML1 benchmark."""
    print(f"[INFO] Setting up MetaWorld ML1 environment for task: {task_name}")
    try:
        benchmark = metaworld.ML1(task_name)
        env = benchmark.train_classes[task_name]()
        task = random.choice(benchmark.train_tasks)
        env.set_task(task)

        # The new API often returns the env within a Gym wrapper, so we handle that
        if hasattr(env, 'env'):
            env = env.env
        env.render_mode = render_mode
        env.camera_name = 'corner'
        return env
    except Exception as e:
        raise RuntimeError(f"Failed to create environment: {e}")

class MetaWorldRGBEvaluator:
    def __init__(self, model_path, checkpoint_path, device="cuda", image_size=336, show_gui=False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.show_gui = show_gui
        print(f"[✓] Device: {self.device}")

        self.model = UnifiedTinyVLAModel(model_path, mode="action").to(self.device)
        self._load_head(checkpoint_path)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        self.action_mean = torch.zeros(4, device=self.device)
        self.action_std = torch.ones(4, device=self.device)

    def _load_head(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"[!] Checkpoint {ckpt_path} not found → using un-trained head")
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        self.model.base_model.embed_out.load_state_dict(ckpt, strict=False)
        self.model.base_model.embed_out = torch.compile(self.model.base_model.embed_out)
        print(f"[✓] Diffusion head loaded from {ckpt_path}")

    def _preprocess_img(self, rgb):
        pil = Image.fromarray(rgb.astype(np.uint8))
        return self.image_processor(pil, return_tensors="pt")["pixel_values"].to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def predict_action(self, rgb, prompt, robot_state=None):
        model_dtype = next(self.model.parameters()).dtype
        img_t = self._preprocess_img(rgb).to(model_dtype)
        tok = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        states = torch.zeros((1, 7), device=self.device) if robot_state is None else torch.tensor(robot_state, device=self.device).unsqueeze(0)
        states = states.to(model_dtype)
        
        try:
            out = self.model.base_model(input_ids=tok.input_ids, attention_mask=tok.attention_mask, images=img_t, states=states, eval=True)
            if isinstance(out, torch.Tensor): act_seq = out
            elif isinstance(out, dict): act_seq = out.get("actions", None)
            else: act_seq = None
            if act_seq is None: return np.zeros(4, dtype=np.float32)
            
            if len(act_seq.shape) >= 3: act = act_seq[0, 0].cpu()
            elif len(act_seq.shape) == 2: act = act_seq[0].cpu()
            else: act = act_seq.cpu()
                
            return (act.float() * self.action_std.cpu().float() + self.action_mean.cpu().float()).numpy()
        except Exception as e:
            print(f"[ERROR] Model prediction failed: {e}")
            return np.zeros(4, dtype=np.float32)

    def collect_trajectory(self, env, prompt, max_steps=150):
        obs, info = env.reset()
        total_reward, frames, actions = 0.0, [], []
        
        print(f"[Collector] Starting trajectory for: {prompt}")

        for t in range(max_steps):
            try:
                rgb = env.render()
            except Exception as e:
                print(f"[!] RGB render failed: {e}")
                rgb = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

            state = obs[:7] if len(obs) >= 7 else None
            act = self.predict_action(rgb, prompt, state)
            actions.append(act) # Store the action
            
            obs, r, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            total_reward += r

            frames.append(rgb)
            
            if t > 0 and t % 10 == 0: print(f"  Step {t}: reward={r:.3f}, total={total_reward:.3f}")
            if done or info.get("success", False):
                if info.get("success", False): print(f"[SUCCESS] @ step {t}!")
                break
                
        return dict(reward=total_reward, steps=t+1, success=info.get("success", False), frames=frames, actions=actions)

    def replay_and_save_video(self, task_name, actions, filename, fps=10):
        """Creates a new env and replays actions to generate a video."""
        print(f"[Video] Replaying {len(actions)} actions to create video...")
        replay_env = setup_env(task_name, render_mode='rgb_array')
        replay_env.reset()
        
        video_frames = []
        for act in actions:
            video_frames.append(replay_env.render())
            replay_env.step(act)
        
        self.save_video(video_frames, filename, fps)

    def save_video(self, frames, filename, fps=10):
        if not frames: return
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        for frame in frames: out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[✓] Video saved: {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lesjie/Llava-Pythia-400M")
    parser.add_argument("--checkpoint_path", default="checkpoints/diff_head_ft.pth")
    parser.add_argument("--task", default="pick-place-v3") # V3 task
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--prompt", default="Pick up the object and place it at the target.")
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()

    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    render_mode = 'rgb_array' # Always use rgb_array for collection
    env = setup_env(args.task, render_mode)
    if env is None: return

    evaluator = MetaWorldRGBEvaluator(args.model_path, args.checkpoint_path)
    
    for ep in range(args.episodes):
        print(f"===== Running Episode {ep+1}/{args.episodes} for task {args.task} =====")
        res = evaluator.collect_trajectory(env, args.prompt, args.max_steps)
        print(f"[Result] success={res['success']} reward={res['reward']:.3f} steps={res['steps']}")

        if args.save_video:
            out_file = f"episode_{ep+1}_{args.task}.mp4"
            evaluator.replay_and_save_video(args.task, res['actions'], out_file)

if __name__ == "__main__":
    main()
