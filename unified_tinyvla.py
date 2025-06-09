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
    This version defines its own action head and denoising loop to ensure
    full control over data types and resolve library-level dtype conflicts.
    """
    def __init__(self, model_path: str, mode: str = "text"):
        super().__init__()
        self.mode = mode
        
        # FIX: We must use float32 to be compatible with the hardcoded bfloat16
        # in the action head's SinusoidalPosEmb module.
        self.model_dtype = torch.float32

        # Load config and EXPLICITLY disable the base model's internal action head
        # This prevents conflicts and ensures this wrapper has full control.
        config = LlavaPythiaConfig.from_pretrained(model_path)
        config.action_head_type = None
        
        # Load the base model, now guaranteed to be just a feature extractor
        self.base_model = LlavaPythiaForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True
        )
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # This wrapper uses its own text decoder
        self.text_decoder = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.base_model.config.vocab_size)
        )

        # Re-introduce this wrapper's own action head and scheduler
        if mode == "action":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            from policy_heads.models import ConditionalUnet1D
            
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='epsilon'
            )
            
            self.action_head = ConditionalUnet1D(
                input_dim=10,  # action dimension
                global_cond_dim=self.base_model.config.hidden_size,
                state_dim=7  # state dimension
            )
            
            self.num_queries = 16
            self.num_inference_timesteps = 10
    
    def forward(self, input_ids, attention_mask=None, images=None, states=None):
        # 1. Get hidden states by calling the core transformer directly
        with torch.no_grad():
            _, _, _, inputs_embeds, _ = self.base_model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images
            )
            outputs = self.base_model.get_model()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1]

        # 2. Generate text logits
        text_logits = self.text_decoder(hidden_states)

        # 3. Run this wrapper's own denoising loop if in action mode
        if self.mode == "action":
            with torch.no_grad():
                # Take last token's hidden state as the global condition.
                # Squeeze the sequence dimension as the UNet expects [B, H].
                global_cond = hidden_states[:, -1]

                if states is not None:
                    states = states.to(dtype=self.model_dtype, device=hidden_states.device)
                    if len(states.shape) == 1:
                        states = states.unsqueeze(0)
                    if len(states.shape) == 3:
                        states = states.squeeze(1)

                B = 1
                Tp = self.num_queries
                action_dim = 10
                naction = torch.randn(
                    (B, Tp, action_dim), 
                    device=hidden_states.device, 
                    dtype=self.model_dtype
                )
                
                # Move scheduler tensors to the correct device
                self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(hidden_states.device)
                self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
                # Ensure timesteps are in long dtype for indexing
                timesteps = self.noise_scheduler.timesteps.to(dtype=torch.long, device=hidden_states.device)

                for k in timesteps:
                    noise_pred = self.action_head(
                        naction, 
                        k, 
                        global_cond=global_cond, 
                        states=states
                    )
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            
            return {
                'text_logits': text_logits,
                'actions': naction,
                'hidden_states': hidden_states
            }
        else:
            return {
                'text_logits': text_logits,
                'hidden_states': hidden_states
            }

def run_inference(image_path, prompt, model_path, mode="vlm"):
    model = UnifiedTinyVLAModel(model_path, mode=mode)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model will be in float32 by default, which is what we need.
    model = model.to(device) 
    
    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    image = Image.open(image_path)
    # Ensure input tensors match the model's float32 dtype
    image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(device=device, dtype=model.model_dtype)
    
    text_tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = text_tokens["input_ids"].to(device)
    attention_mask = text_tokens["attention_mask"].to(device)
    
    states = None
    if mode == "action":
        stats_path = os.path.join(model_path, "norm_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                norm_stats = pickle.load(f)
            states = torch.zeros((1, 7), device=device, dtype=model.model_dtype)
            qpos_mean = torch.tensor(norm_stats["qpos_mean"], device=device, dtype=model.model_dtype)
            qpos_std = torch.tensor(norm_stats["qpos_std"], device=device, dtype=model.model_dtype)
            states = (states - qpos_mean) / qpos_std
        else:
            print("Warning: norm_stats.pkl not found, using unnormalized states")
            states = torch.zeros((1, 7), device=device, dtype=model.model_dtype)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            states=states
        )
    
    if mode == "action":
        text_ids = torch.argmax(outputs['text_logits'], dim=-1)
        text = tokenizer.decode(text_ids[0], skip_special_tokens=True)
        actions = outputs['actions']
        return {'text': text, 'actions': actions, 'chain_of_thought': text}
    else:
        text_ids = torch.argmax(outputs['text_logits'], dim=-1)
        text = tokenizer.decode(text_ids[0], skip_special_tokens=True)
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
