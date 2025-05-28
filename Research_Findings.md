# This is the markdown file for research findings of this repository
## There shall be certain findings for each script
### In script ```python run_paligemma2.py```` :

#### Observations
Where an Image with white board and eraser is presented from the Gemini Robotics demo video (https://www.youtube.com/watch?v=07uzLyhCqcQ) and the text prompting the vlm as robotic agent to produce CoT on how to erase the board, the VLM failed to produce meaningful text and simply repeated the same text prompt.

#### Script Takeaway
- We discovered that Google's PaliGemma2 VLM, while generally preserves image captioning ability, lacks fundamental spatial awaress possibly due to lack of dataset on Robotic Manipulation.
- Need to **gather robotics finetuning dataset** (OpenX subset?) or **download pretrained VLM backbone** (LlaVA Backbone from TinyVLA)