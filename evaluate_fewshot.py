import torch, numpy as np, argparse
from unified_tinyvla import UnifiedTinyVLAModel
from metaworld import ML1
from transformers import AutoTokenizer
import torchvision.transforms as T

tf = T.Compose([T.Resize((224,224)), T.ToTensor()])

def rollout(env, model, tok_prompt, steps=150):
    img = env.render(offscreen=True, mode="rgb_array")
    img_t = tf(img).unsqueeze(0).cuda()
    fused = model.vlm_encode(img_t, **tok_prompt)
    act = model.action_head.sample(fused)[0].cpu().numpy()
    obs, rew, done, info = env.step(act)
    return rew, done, info

def eval_task(task, model, tokenizer, shots=1, episodes=20):
    ml1 = ML1(task); env_cls = ml1.train_classes[task]
    env = env_cls(); env.set_task(ml1.train_tasks[0])
    prompt = f"{task} task. Perform it."
    tok = tokenizer(prompt, return_tensors="pt").to("cuda")

    successes = 0
    for ep in range(episodes):
        env.reset(); ep_done=False
        for _ in range(150):
            _, done, info = rollout(env, model, tok)
            if done:
                ep_done=True; break
        successes += int(ep_done or info.get("success", False))
    return successes / episodes

def main(args):
    model = UnifiedTinyVLAModel(args.model_path, mode="action").cuda().eval()
    model.action_head.load_state_dict(torch.load(args.head_ckpt))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    acc = eval_task(args.task, model, tokenizer,
                    shots=args.shots, episodes=args.episodes)
    print(f"{args.task}  success-rate: {acc:.2%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick-place-v1")
    ap.add_argument("--model_path", default="VLM_weights/Llava-Pythia-400M")
    ap.add_argument("--head_ckpt", default="checkpoints/diff_head_ft.pth")
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args(); main(args)
