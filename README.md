# This is a test repo for practicing Transformers, VLMs, and VLAs using the Google GEMMA3-1B-it Model and TinyVLA
## This repository assumes Ubuntu 22.04 for Unix-like shells and further SDE tools in use
## 🧠 Gemma-VLM (Vision-Language Model)

A hand-built modular vision-language model (VLM) pipeline combining:
- **SigLIP** as image encoder (`google/siglip-base-patch16-224`)
- **Linear projection** from 768 → 2048
- **Gemma 3 - 1B** as language decoder (`google/gemma-3-1b-it`)

### 🔧 Installation

#### 1. Clone and create a Conda environment
```bash
git clone https://github.com/HanshangZhu/gemma-vlm
cd gemma-vlm
conda create -n gemma-vlm-test python=3.10
conda activate gemma-vlm-test
pip install -r requirements.txt
```

#### 2. HuggingFace Cache Management
**For inference** (simply calling models and generate output with prompts) ,we utilise *HuggingFace Transformers* Library which **downloads and caches model weights locally**. Given the goal of this repo, the process involves trialing with multiple VLM backbone and this process could be unresourceful and rather inefficient.

Hence, we recommend the following commands if you wish to free up space:
- *To see which models are currently downloaded*:

```bash
ls ~/.cache/huggingface/hub/
```

- *Example Output*:

```bash
models--google--gemma-1.1-2b-it
models--google--siglip-base-patch16-224
models--google--paligemma-3b-pt-224
```

- *Selecting a model you wish to delete*
```bash
rm -rf ~/.cache/huggingface/hub/models--google--siglip-base-patch16-224
```

## 🤖 TinyVLA (Vision-Language-Action Model)

This repository includes a complete implementation of TinyVLA for robotic manipulation tasks, featuring:
- **Unified TinyVLA Model** with diffusion-based action prediction
- **Training pipeline** for fine-tuning on robotic datasets  
- **MetaWorld evaluation** with RGB rendering for real-world testing

### 🏋️ Training

#### 1. Prepare Your Dataset
Place your dataset in the appropriate format. The training script expects:
- **Images**: RGB observations from robot cameras
- **Actions**: 4D action vectors (x, y, z, gripper)
- **Prompts**: Task descriptions in natural language

#### 2. Train the Diffusion Head
```bash
# Train TinyVLA on short MetaWorld dataset
python train_tinyvla_policy.py \
    --data_root ./datasets/short-metaworld \
    --model_path VLM_weights/Llava-Pythia-400M \
    --tasks pick-place-v2,door-open-v2,drawer-open-v2 \
    --epochs 10 \
    --bs 16 \
    --lr 3e-4 \
    --out_dir checkpoints
```

The training process:
1. **Loads pre-trained VLM weights** from `model_path`
2. **Freezes base model parameters** (only trains the diffusion head)
3. **Uses diffusion loss** to learn action prediction from vision and language
4. **Saves trained checkpoint** to `checkpoints/diff_head_ft.pth`

#### 3. Training Progress
The script will output training progress:
```
epoch 00 | loss 0.1234 | elapsed 45.2s
epoch 01 | loss 0.0987 | elapsed 89.7s
...
✔ saved diffusion head to checkpoints
```

### 🎯 MetaWorld RGB Evaluation

After training, evaluate your model on RGB-rendered MetaWorld tasks to test real-world performance.

#### 1. Test MetaWorld Setup
```bash
# Activate the tinyvla environment and test MetaWorld
conda activate tinyvla
python -c "import metaworld; print('✓ MetaWorld installed successfully!')"

# Test a simple environment
python -c "
import metaworld
import random
ml1 = metaworld.ML1('pick-place-v2')
env = ml1.train_classes['pick-place-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)
obs = env.reset()
print(f'✓ Environment works! Obs shape: {obs.shape}')
"
```

#### 2. Run Evaluation
```bash
# Basic evaluation on pick-place task
python eval_metaworld_rgb.py \
    --task pick-place-v2 \
    --episodes 5 \
    --max_steps 150

# Advanced evaluation with video recording
python eval_metaworld_rgb.py \
    --task pick-place-v2 \
    --episodes 10 \
    --max_steps 200 \
    --save_video \
    --prompt "Pick up the object and place it at the target location"

# Evaluate on different tasks
python eval_metaworld_rgb.py --task door-open-v2 --prompt "Open the door"
python eval_metaworld_rgb.py --task drawer-open-v2 --prompt "Open the drawer"
python eval_metaworld_rgb.py --task button-press-v2 --prompt "Press the button"
```

#### 4. Evaluation Output
The script provides comprehensive evaluation metrics:
```
🚀 Starting evaluation on pick-place-v2
Task prompt: 'Pick up the object and place it at the target location'
Episodes: 5
Max steps per episode: 150

==================================================
Episode 1/5
==================================================
--- Starting episode with prompt: 'Pick up the object and place it at the target location' ---
Step 0: action=[-0.123, 0.456, 0.789, 1.000], reward=0.000, total_reward=0.000
Step 10: action=[0.234, -0.567, 0.012, 0.500], reward=0.100, total_reward=1.250
...
Episode finished at step 45: success=True

==================================================
EVALUATION SUMMARY
==================================================
Task: pick-place-v2
Episodes: 5
Success Rate: 80.0% (4/5)
Average Reward: 15.234
Average Steps: 67.2

Detailed Results:
  Episode 1: ✓ Reward: 18.456, Steps: 45
  Episode 2: ✓ Reward: 16.789, Steps: 89
  Episode 3: ✗ Reward: 8.123, Steps: 150
  Episode 4: ✓ Reward: 19.567, Steps: 52
  Episode 5: ✓ Reward: 13.234, Steps: 100
```

#### 5. Available MetaWorld Tasks
The evaluation supports all MetaWorld v2 tasks:
- **pick-place-v2**: Pick up object and place at target
- **door-open-v2**: Open a door by turning the handle
- **drawer-open-v2**: Pull open a drawer
- **button-press-v2**: Press a button
- **reach-v2**: Reach to a target position
- **push-v2**: Push object to target location
- **window-open-v2**: Slide open a window
- **sweep-v2**: Sweep object to target area
- And 42+ more manipulation tasks...

#### 6. Video Recording
When using `--save_video`, the script saves MP4 videos of each episode:
```
✓ Saved video to episode_1_pick-place-v2.mp4
✓ Saved video to episode_2_pick-place-v2.mp4
...
```

### 🔧 Architecture Details

#### UnifiedTinyVLAModel
The core model combines:
- **Base VLM**: LlavaPythiaForCausalLM with vision and language understanding
- **Diffusion Head**: ConditionalUnet1D for action sequence prediction
- **Inference Mode**: Iterative denoising from random noise to actions

#### Diffusion Training Process
1. **Forward Process**: Add noise to ground-truth actions
2. **Denoising Network**: Learn to predict noise at each timestep  
3. **Loss Function**: MSE between predicted and actual noise
4. **Inference**: Start with noise, iteratively denoise to get actions

#### Key Features
- **Vision-Language Understanding**: Processes RGB images and text prompts
- **Action Sequence Prediction**: Outputs 20-step action sequences
- **Diffusion-based Generation**: Robust and high-quality action prediction
- **Real-world Transfer**: Trained on simulation, evaluated on RGB environments

### 📊 Performance Metrics

The evaluation provides several key metrics:
- **Success Rate**: Percentage of episodes that complete the task successfully
- **Average Reward**: Mean cumulative reward across episodes
- **Average Steps**: Mean number of steps to complete (or timeout)
- **Individual Results**: Per-episode breakdown with success/failure status

### 🛠️ Troubleshooting

#### Common Issues
1. **MuJoCo Installation**: Make sure MuJoCo 2.1+ is properly installed
2. **Headless Rendering**: Use `xvfb-run -a` prefix for servers without displays
3. **GPU Memory**: Reduce batch size if encountering CUDA OOM errors
4. **MetaWorld Issues**: If MetaWorld import fails, reinstall with:
   ```bash
   pip install git+https://github.com/Farama-Foundation/Metaworld.git
   ```

#### Performance Tips
- Use `--max_steps 200` for complex tasks that need more time
- Try different prompts to see how language affects performance  
- Use `--save_video` to visually debug model behavior
- Test on simpler tasks like `reach-v2` before complex manipulation

### 🎯 Next Steps
- **Multi-task Training**: Train on multiple MetaWorld tasks simultaneously
- **Real Robot Transfer**: Deploy trained models on physical robot systems
- **Advanced Prompting**: Experiment with more detailed task descriptions
- **Hyperparameter Tuning**: Optimize learning rates and model architectures

### Datasets
*short-MetaWorld*
https://connecthkuhk-my.sharepoint.com/personal/liangzx_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fliangzx%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fshort%2DMetaWorld&ga=1

