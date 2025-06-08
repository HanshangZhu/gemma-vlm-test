import torch
from llava_pythia.conversation import conv_templates
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image
import argparse
from huggingface_hub import snapshot_download
import os
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).parent.absolute()
MODELS_DIR = ROOT_DIR / "VLM_weights"
TEST_IMGS_DIR = ROOT_DIR / "test_imgs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
TEST_IMGS_DIR.mkdir(exist_ok=True)

class VLMInference:
    def __init__(self, model_path, model_base=None, conv_mode="llava_v1"):
        """
        Initialize the VLM inference model.
        
        Args:
            model_path (str): Path to the model weights
            model_base (str, optional): Base model path for LoRA models
            conv_mode (str): Conversation template mode
        """
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        
        # Load model and tokenizer
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_path, False, False
        )
        
        # Initialize conversation template
        self.conv = conv_templates[conv_mode].copy()
        
    def process_image(self, image_path):
        """
        Process an image for model input.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            torch.Tensor: Processed image tensor
        """
        image = Image.open(image_path)
        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(
            image, 
            return_tensors='pt', 
            do_normalize=True, 
            do_rescale=False,
            do_center_crop=False
        )['pixel_values']
        return image_tensor.to(self.model.device, dtype=self.model.dtype)
    
    def expand2square(self, pil_img, background_color):
        """
        Expand image to square shape with padding.
        
        Args:
            pil_img (PIL.Image): Input image
            background_color (tuple): Background color for padding
            
        Returns:
            PIL.Image: Square image with padding
        """
        width, height = pil_img.size
        max_dim = max(width, height)
        new_img = Image.new('RGB', (max_dim, max_dim), background_color)
        
        if width == height:
            return pil_img
        elif width > height:
            offset = (max_dim - height) // 2
            new_img.paste(pil_img, (0, offset))
        else:
            offset = (max_dim - width) // 2
            new_img.paste(pil_img, (offset, 0))
            
        return new_img
    
    def generate_response(self, image_path, prompt):
        """
        Generate a response for the given image and prompt.
        
        Args:
            image_path (str): Path to the input image
            prompt (str): Text prompt for the model
            
        Returns:
            str: Generated response
        """
        # Process image
        image_tensor = self.process_image(image_path)
        
        # Prepare conversation
        self.conv = conv_templates[self.conv_mode].copy()
        inp = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"
        
        # Tokenize input
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)
        
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=512,
                use_cache=True,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            output_ids[0], 
            skip_special_tokens=True
        )
        
        return response

def download_model(model_name="400M"):
    """
    Download the model from Hugging Face if it doesn't exist locally.
    
    Args:
        model_name (str): Model size identifier ("400M", "700M", or "1.3B")
    
    Returns:
        Path: Path to the downloaded model
    """
    model_map = {
        "400M": "lesjie/Llava-Pythia-400M",
        "700M": "lesjie/Llava-Pythia-700M",
        "1.3B": "lesjie/Llava-Pythia-1.3B"
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model size must be one of {list(model_map.keys())}")
    
    model_dir = MODELS_DIR / f"Llava-Pythia-{model_name}"
    
    if not model_dir.exists():
        print(f"Downloading {model_name} model...")
        snapshot_download(
            repo_id=model_map[model_name],
            local_dir=model_dir
        )
        print(f"Model downloaded to {model_dir}")
    else:
        print(f"Model already exists at {model_dir}")
    
    return model_dir

def main():
    parser = argparse.ArgumentParser(description='VLM Inference Script')
    parser.add_argument('--model_size', type=str, default="400M", 
                      choices=["400M", "700M", "1.3B"],
                      help='Size of the model to use')
    parser.add_argument('--image_path', type=str, required=True, 
                      help='Path to the input image')
    parser.add_argument('--prompt', type=str, required=True, 
                      help='Text prompt for the model')
    parser.add_argument('--conv_mode', type=str, default='llava_v1', 
                      help='Conversation template mode')
    
    args = parser.parse_args()
    
    # Download model if needed
    model_path = download_model(args.model_size)
    
    # Initialize VLM
    vlm = VLMInference(
        model_path=str(MODELS_DIR / f"Llava-Pythia-{args.model_size}"),
        model_base=None,
        conv_mode=args.conv_mode
    )
    
    # Generate response
    response = vlm.generate_response(args.image_path, args.prompt)
    print("\nGenerated Response:")
    print(response)

if __name__ == "__main__":
    main() 
    # python vlm_inference.py --model_size 400M --image_path test_imgs/your_image.jpg --prompt "What do you see in this image?" 