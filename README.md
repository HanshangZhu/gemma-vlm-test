# This is a test repo for practicing Transformers, VLMs, and VLAs using the Google GEMMA3-1B-it Model
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
