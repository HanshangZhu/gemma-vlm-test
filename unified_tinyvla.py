# unified_tinyvla.py
import os
import sys
sys.path.append('/home/hz/gemma-vlm-test/TinyVLA')
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Optional, Union, Tuple
from transformers import (
    AutoTokenizer, 
    CLIPImageProcessor,
    AutoConfig
)
from llava_pythia.model.language_model.pythia.llava_pythia import (
    LlavaPythiaForCausalLM,
    LlavaPythiaConfig
)
import pickle

class UnifiedTinyVLAModel(nn.Module):
    """
    A wrapper model that uses a pre-trained LlavaPythiaForCausalLM base model.
    This version uses the base model's built-in diffusion head for action prediction.
    """
    def __init__(self, model_path: str, mode: str = "text"):
        super().__init__()
        self.mode = mode
        
        # Use float32 for the diffusion process
        self.model_dtype = torch.float32

        # Load config and set up the base model with diffusion head
        config = LlavaPythiaConfig.from_pretrained(model_path)
        config.action_head_type = 'droid_diffusion'  # Use the built-in diffusion head
        config.action_dim = 4  # (x, y, z, gripper)
        config.state_dim = 7  # state dimension
        config.chunk_size = 20  # sequence length for action prediction
        
        # Load the base model with diffusion head
        self.base_model = LlavaPythiaForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True
        )
        
        # Freeze base model parameters except for the action head
        for name, param in self.base_model.named_parameters():
            if 'embed_out' not in name:  # Only train the action head
                param.requires_grad = False
            
        # This wrapper uses its own text decoder
        self.text_decoder = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.base_model.config.vocab_size)
        )
    
    def forward(self, input_ids, attention_mask, images, states=None, actions=None, is_pad=None):
        B = images.shape[0]  # Get actual batch size from images
        
        # Create dummy is_pad if not provided but actions are provided (training mode)
        if actions is not None and is_pad is None:
            # Create is_pad tensor - assume no padding for now
            is_pad = torch.zeros(actions.shape[:-1], dtype=torch.bool, device=actions.device)
        
        # Get predictions from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            states=states,
            actions=actions,  # Pass actions to base model for training
            is_pad=is_pad,    # Pass is_pad for training
            return_dict=True
        )
        
        # Extract loss from the outputs
        # During training, the diffusion head returns loss in outputs.loss
        loss = None
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            # If outputs.loss is a dict (from diffusion head), extract the 'loss' key
            if isinstance(outputs.loss, dict) and 'loss' in outputs.loss:
                loss = outputs.loss['loss']
            else:
                loss = outputs.loss
        
        # The base model's diffusion head will handle action prediction
        return {
            'text_logits': outputs.logits if hasattr(outputs, 'logits') else None,
            'actions': outputs.actions if hasattr(outputs, 'actions') else None,
            'loss': loss
        }

def run_inference(image_path, prompt, model_path, mode="vlm"):
    model = UnifiedTinyVLAModel(model_path, mode=mode)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert model to float32
    model = model.to(device, dtype=torch.float32)
    
    # Load the trained diffusion head checkpoint
    checkpoint_path = "checkpoints/diff_head_ft.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Remove _orig_mod. prefix if present
        new_checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
        model.base_model.embed_out.load_state_dict(new_checkpoint)
        print(f"✓ Loaded diffusion head checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found")
    
    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    image = Image.open(image_path)
    # Convert image tensor to float32
    image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(
        device=device, dtype=torch.float32
    )
    
    text_tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = text_tokens["input_ids"].to(device)
    attention_mask = text_tokens["attention_mask"].to(device)
    
    states = None
    if mode == "action":
        stats_path = os.path.join(model_path, "norm_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                norm_stats = pickle.load(f)
            states = torch.zeros((1, 7), device=device, dtype=torch.float32)
            qpos_mean = torch.tensor(norm_stats["qpos_mean"], device=device, dtype=torch.float32)
            qpos_std = torch.tensor(norm_stats["qpos_std"], device=device, dtype=torch.float32)
            states = (states - qpos_mean) / qpos_std
        else:
            print("Warning: norm_stats.pkl not found, using unnormalized states")
            states = torch.zeros((1, 7), device=device, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            states=states
        )
    
    if mode == "action":
        # Handle case where text_logits might be None
        if outputs['text_logits'] is not None:
            text_ids = torch.argmax(outputs['text_logits'], dim=-1)
            text = tokenizer.decode(text_ids[0], skip_special_tokens=True)
        else:
            text = "No text generated"
        
        actions = outputs['actions']
        return {'text': text, 'actions': actions, 'chain_of_thought': text}
    else:
        if outputs['text_logits'] is not None:
            text_ids = torch.argmax(outputs['text_logits'], dim=-1)
            text = tokenizer.decode(text_ids[0], skip_special_tokens=True)
        else:
            text = "No text generated"
        return {'text': text}

if __name__ == '__main__':
    try:
        print("\n=== Starting TinyVLA Inference ===")
        outputs = run_inference(
            image_path="test_imgs/task_clean_whiteboard.png",
            prompt="What do you see in this image?",
            model_path="VLM_weights/Llava-Pythia-400M",
            mode="action"
        )

        print("\n--- Final Outputs ---")
        print("Chain of Thought:", outputs['chain_of_thought'])
        if 'actions' in outputs:
            print("Final Actions shape:", outputs['actions'].shape)
            print("First action sequence:", outputs['actions'][0])
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
