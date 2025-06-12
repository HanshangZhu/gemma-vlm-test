#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

# Try different rendering modes
render_modes = ['osmesa', 'egl', 'glfw']

for mode in render_modes:
    print(f"\n=== Testing {mode} rendering ===")
    
    # Set environment
    os.environ['MUJOCO_GL'] = mode
    if mode == 'osmesa':
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    
    try:
        # Import after setting environment
        import metaworld
        from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS
        
        # Create environment
        env_cls = ALL_ENVIRONMENTS['pick-place-v1']
        env = env_cls(intervention_id='000', experiment_id='_exp0_seed=0', apply_mod=False)
        
        # Reset and try to render
        obs = env.reset()
        print(f"Environment created successfully")
        
        # Try rendering
        try:
            img = env.render(mode='rgb_array', width=224, height=224)
            if img is not None:
                print(f"✅ Rendering successful! Image shape: {img.shape}")
                print(f"   Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.2f}")
                
                # Save sample image
                plt.figure(figsize=(6, 6))
                plt.imshow(img)
                plt.title(f'MetaWorld Rendering - {mode}')
                plt.axis('off')
                plt.savefig(f'test_render_{mode}.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved sample image: test_render_{mode}.png")
                
                # Check if it's not just a black image
                if img.max() > 10:  # If max pixel value > 10, probably has content
                    print(f"✅ Image has actual content!")
                    break
                else:
                    print(f"⚠️  Image appears to be mostly black/empty")
            else:
                print(f"❌ Rendering returned None")
                
        except Exception as e:
            print(f"❌ Rendering failed: {e}")
            
        env.close()
        
    except Exception as e:
        print(f"❌ Failed to create environment with {mode}: {e}")

print(f"\n=== Test completed ===") 