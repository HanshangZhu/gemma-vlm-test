# ===================================================================
# DeepSpeed Integration Patch for train_lora.py
# 
# This file shows the key changes needed to add DeepSpeed support.
# Apply these changes to your existing train_lora.py
# ===================================================================

import argparse
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

# ============= 1. ADD COMMAND LINE ARGUMENTS =============
def parse_args():
    """Add DeepSpeed arguments to existing argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    
    # DeepSpeed arguments
    parser.add_argument("--deepspeed", type=str, help="DeepSpeed config JSON file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # Add other DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()

# ============= 2. MODIFY TRAINER INITIALIZATION =============
class Trainer:
    def __init__(self, config: TrainingConfig, args=None):
        self.config = config
        self.args = args  # Store command line args
        
        # Set device - DeepSpeed will handle multi-GPU
        if args and args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device(f"cuda:{args.local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"✅ Using device: {self.device}")
        
        # Initialize other components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.data_loader = None
        self.stats = None
        self.best_loss = float('inf')
        
        # DeepSpeed engine (will be set later)
        self.model_engine = None

    # ============= 3. MODIFY MODEL SETUP FOR DEEPSPEED =============
    def _setup_model_and_tokenizer(self):
        print("🔄 Loading model and tokenizer...")
        
        # Load config and model as before
        patched_cfg = LlavaPythiaConfig.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        
        # ... existing model loading code ...
        
        # Move model to device (DeepSpeed will handle distribution)
        self.model = self.model.to(self.device)
        
        print(f"✅ Model loaded: {type(self.model).__name__}")

    # ============= 4. REPLACE OPTIMIZER SETUP WITH DEEPSPEED =============
    def _setup_deepspeed(self):
        """Initialize DeepSpeed engine"""
        if not self.args or not self.args.deepspeed:
            # Fallback to regular training
            self._setup_optimizer()
            return
            
        print("🔧 Initializing DeepSpeed...")
        
        # DeepSpeed handles optimizer, scheduler, and model wrapping
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            args=self.args,
            model=self.model,
            model_parameters=self.model.parameters(),
        )
        
        # Update device to match DeepSpeed
        self.device = self.model_engine.local_rank
        
        print(f"✅ DeepSpeed initialized with ZeRO stage {self.model_engine.zero_optimization_stage()}")

    # ============= 5. MODIFY TRAINING LOOP =============
    def train(self):
        self._setup()
        
        # Setup DeepSpeed instead of regular optimizer
        self._setup_deepspeed()
        
        # Set training mode
        if self.model_engine:
            self.model_engine.train()
        else:
            self.model.train()
            
        global_step = 0
        
        print(f"🚀 Starting training with DeepSpeed: {len(self.data_loader)} batches/epoch")
        
        while global_step < self.config.max_steps:
            for batch_idx, batch in enumerate(self.data_loader):
                if global_step >= self.config.max_steps:
                    break
                    
                # Move batch to device
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch[key] = value.to(self.device)
                
                # Forward pass
                if self.model_engine:
                    # DeepSpeed handles mixed precision automatically
                    outputs = self.model_engine(**batch)
                else:
                    # Regular training fallback
                    with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
                        outputs = self.model(**batch)
                
                # Get loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    # Handle other loss computation
                    logits = outputs.logits
                    labels = batch["labels"]
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Backward pass
                if self.model_engine:
                    # DeepSpeed handles backward, gradient accumulation, and optimizer step
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                else:
                    # Regular training fallback
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                # Logging
                if global_step % 100 == 0:
                    print(f"Step {global_step}: Loss = {loss.item():.4f}")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0 and global_step > 0:
                    self.save_checkpoint_deepspeed(global_step, loss.item())
                
                global_step += 1

    # ============= 6. MODIFY CHECKPOINT SAVING =============
    def save_checkpoint_deepspeed(self, step, loss_value):
        """Save checkpoint with DeepSpeed support"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"step_{step}")
        
        if self.model_engine:
            # DeepSpeed checkpoint saving
            self.model_engine.save_checkpoint(checkpoint_dir, tag=f"step_{step}")
            
            # Also save the consolidated FP32 model for easy loading
            fp32_model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if os.path.exists(os.path.join(checkpoint_dir, f"step_{step}")):
                state_dict = get_fp32_state_dict_from_zero_checkpoint(
                    os.path.join(checkpoint_dir, f"step_{step}")
                )
                torch.save(state_dict, fp32_model_path)
                print(f"💾 Saved FP32 model to {fp32_model_path}")
        else:
            # Regular checkpoint saving
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.model.save_pretrained(checkpoint_dir)
        
        print(f"💾 Saved checkpoint at step {step} to {checkpoint_dir}")

# ============= 7. MODIFY MAIN FUNCTION =============
def main():
    """Modified main function with DeepSpeed support"""
    args = parse_args()
    
    # Load config from YAML
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override output_dir if provided via command line
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir
    
    config = TrainingConfig(**config_dict)
    
    # Initialize distributed training if using DeepSpeed
    if args.deepspeed:
        deepspeed.init_distributed()
    
    # Create trainer with args
    trainer = Trainer(config, args)
    trainer.train()

if __name__ == "__main__":
    main()

# ===================================================================
# PLACEHOLDERS CREATED IN THIS PATCH:
# 
# None - this patch doesn't create placeholders, but note that the
# job script created these placeholders:
# 
# 1. YOUR_UCL_EMAIL@ucl.ac.uk - Replace with your actual UCL email
# 2. NCCL_SOCKET_IFNAME=eth0 - May need to change to ib0 or other 
#    network interface name depending on UCL's cluster setup
# =================================================================== 