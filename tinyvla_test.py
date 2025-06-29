#!/usr/bin/env python3
import torch
from PIL import Image
import numpy as np
import sys
import os
import pickle
from transformers import AutoTokenizer

# Add TinyVLA paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'TinyVLA')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'TinyVLA', 'llava-pythia')))

# Imports from eval_real_franka.py
from llava_pythia.conversation import conv_templates
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_pythia.model import * # Wildcard import to resolve scope issues
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaForCausalLM

class llava_pythia_act_policy:
    """
    Policy class for Llava-Pythia action generation.
    Copied directly from TinyVLA/eval_real_franka.py to ensure correct scope.
    """
    def __init__(self, policy_config):
        super(llava_pythia_act_policy).__init__()
        # Ann: Manually defining config here to resolve NameError
        # This is the line that was causing issues before.
        global LlavaPythiaConfig
        from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig

        self.load_policy(policy_config)

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_path = policy_config["model_path"]
        model_name = get_model_name_from_path(model_path)

        # --- Corrected Model Loading Sequence ---
        # 1. Load config and tokenizer from path
        print("✅ Loading config and tokenizer...")
        config = LlavaPythiaConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        # 2. **Modify the config BEFORE loading the model**
        print("✅ Applying necessary config modifications...")
        config.action_head_type = 'droid_diffusion'
        config.action_dim = 10
        config.state_dim = 9
        config.chunk_size = 20
        config.concat = 'token_cat' # The critical fix!
        config.mm_use_im_start_end = True # From builder.py
        
        # 3. Load the model with the modified config
        print("✅ Loading model with modified config...")
        self.policy = LlavaPythiaForCausalLM.from_pretrained(
            model_path,
            config=config,
            use_safetensors=True,
            torch_dtype=torch.float16
        ).to("cuda")

        # 4. Load the image processor
        print("✅ Loading image processor...")
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)

        # 5. Add special tokens
        from llava_pythia.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.context_len = getattr(self.policy.config, "max_sequence_length", 2048)
        # --- End Corrected Loading ---

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()

        if len(curr_image.shape) == 5: # (1, 2, C, H, W) -> (2, C, H, W)
            curr_image = curr_image.squeeze(0)

        image, image_r = torch.chunk(curr_image, 2, dim=0)

        # Preprocess images
        image_tensor = self.preprocess_single_image(image)
        image_tensor_r = self.preprocess_single_image(image_r)

        # Prepare prompt
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + raw_lang
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt() + " <|endoftext|>"

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)

        return dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensor,
            images_r=image_tensor_r,
            states=states
        )

    def preprocess_single_image(self, image_tensor):
        """ Helper to preprocess one image tensor """
        # The original code used a custom 'expand2square', but CLIP processor handles it.
        # It expects a PIL image or a numpy array, so we convert from tensor.
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        processed = self.image_processor.preprocess(image_pil, return_tensors='pt')['pixel_values']
        return processed.to(self.policy.device, dtype=self.policy.dtype)

def main():
    """
    Main function to run inference with the self-contained TinyVLA policy.
    """
    print("🤖 Initializing TinyVLA Policy...")

    policy_config = {
        'model_path': 'VLM_weights/Llava-Pythia-400M',
        'model_base': None,
        'conv_mode': 'pythia',
    }

    try:
        # 1. Load the policy
        policy = llava_pythia_act_policy(policy_config=policy_config)
        policy.policy.eval()
        device = policy.policy.device
        model_dtype = policy.policy.dtype
        print(f"✅ Policy loaded on device: {device}, dtype: {model_dtype}")

        # 2. Prepare dummy inputs
        print("🖼️  Preparing dummy inputs...")
        image = Image.new('RGB', (224, 224), 'purple')
        image_np = np.array(image).transpose(2, 0, 1) # HWC -> CHW
        images_np = np.stack([image_np, image_np], axis=0) # [2, C, H, W]
        images_tensor = torch.from_numpy(images_np).unsqueeze(0).float() / 255.0

        robot_state = torch.zeros((1, 9), device=device, dtype=model_dtype)
        instruction = "Grasp the red block."
        print(f"   Inputs ready for instruction: '{instruction}'")

        # 3. Pre-process the inputs
        data_dict = policy.process_batch_to_llava(images_tensor, robot_state, instruction)
        print("✅ Input data pre-processed.")
        
        # 4. Run inference
        print("🚀 Running inference...")
        with torch.no_grad():
            # --- Correct Two-Step Inference ---
            # 1. Get the global_cond from the VLM's hidden states
            vlm_output = policy.policy.forward(**data_dict, output_hidden_states=True, eval=True)
            global_cond = vlm_output.hidden_states[-1][:, -1]

            # 2. Use the global_cond to get actions from the action head
            # The action head is the 'embed_out' layer of the policy model.
            noisy_actions = torch.randn(
                (1, policy.policy.config.chunk_size, policy.policy.config.action_dim),
                device=device, dtype=model_dtype
            )
            timestep = torch.tensor([0], device=device, dtype=model_dtype)
            
            actions = policy.policy.embed_out(
                noisy_actions,
                timestep,
                global_cond=global_cond
            )
            output = {'actions': actions} # Manually create the output dict

        # 5. Extract and print actions
        if 'actions' in output and output['actions'] is not None:
            actions = output['actions']
            print("🎉 SUCCESS! Inference completed.")
            print(f"   Predicted actions shape: {actions.shape}")
            print(f"   First action sequence: {actions[0]}")
        else:
            print("❌ ERROR: 'actions' key not found or is None in model output.")
            print("   Available keys:", list(output.keys()))
            if 'logits' in output:
                # If we got logits, let's see what the model is "saying"
                text_ids = torch.argmax(output['logits'], dim=-1)
                text_output = policy.tokenizer.decode(text_ids[0], skip_special_tokens=True)
                print(f"   Raw text output: '{text_output.strip()}'")


    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Add an import for CLIPImageProcessor which is now used in the rewritten load_policy
    from transformers import CLIPImageProcessor
    main() 