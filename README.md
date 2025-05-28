# This is a test repo for practicing Transformers, VLMs, and VLAs using the Google GEMMA3-1B-it Model
## This repository assumes Ubuntu 22.04 for Unix-like shells and further SDE tools in use
## 🧠 Gemma-VLM (Vision-Language Model)

A hand-built modular vision-language model (VLM) pipeline combining:
- **SigLIP** as image encoder (`google/siglip-base-patch16-224`)
- **Linear projection** from 768 → 2048
- **Gemma 2B** as language decoder (`google/gemma-1.1-2b-it`)

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
For inference (simply calling models and generate output with prompts) ,we utilise HuggingFace Transformers Library which downloads and caches model weights locally in your system. Given the goal of this repo, the process involves trialing with multiple VLM backbone and this process *could be cumbersome*

Hence, we recommend the following commands if you wish to free up space:
*To see which models are currently downloaded*:

```bash
ls ~/.cache/huggingface/hub/
```

Example Output:

```bash
models--google--gemma-1.1-2b-it
models--google--siglip-base-patch16-224
models--google--paligemma-3b-pt-224
```

Selecting a model you wish to delete
```bash
rm -rf ~/.cache/huggingface/hub/models--google--siglip-base-patch16-224
```

