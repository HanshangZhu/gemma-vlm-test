#!/usr/bin/env python3
"""
Test script for the Short-MetaWorld dataset loader
Tests the loader with the current local dataset structure
"""

import sys
from pathlib import Path

# Add the loader code inline for testing
exec(open('comprehensive_metaworld_backup.py').read().split('def create_dataset_loader():')[1].split('return loader_code')[0].split("'''")[1])

def test_loader():
    """Test the dataset loader with current structure"""
    print("🧪 Testing Short-MetaWorld Dataset Loader")
    print("=" * 50)
    
    # Test with current structure
    data_root = "datasets"
    
    try:
        # Load dataset
        print(f"📁 Loading dataset from: {data_root}")
        dataset = ShortMetaWorldDataset(data_root)
        
        # Get statistics
        stats = dataset.get_dataset_stats()
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total steps: {stats['total_steps']}")
        print(f"   Number of tasks: {stats['num_tasks']}")
        print(f"   Available tasks: {stats['tasks'][:5]}{'...' if len(stats['tasks']) > 5 else ''}")
        
        # Test a sample
        if len(dataset) > 0:
            print(f"\n📋 Testing sample access...")
            sample = dataset[0]
            print(f"   ✅ Image shape: {sample['image'].shape}")
            print(f"   ✅ State shape: {sample['state'].shape}")
            print(f"   ✅ Action shape: {sample['action'].shape}")
            print(f"   ✅ Task: {sample['task_name']}")
            print(f"   ✅ Prompt: {sample['prompt'][:80]}...")
            
            # Test multiple samples
            print(f"\n🔄 Testing multiple samples...")
            for i in range(min(3, len(dataset))):
                s = dataset[i]
                print(f"   Sample {i}: Task={s['task_name']}, Traj={s['trajectory_id']}, Step={s['step_id']}")
        
        # Test task info
        if dataset.tasks:
            task = dataset.tasks[0]
            task_info = dataset.get_task_info(task)
            print(f"\n📖 Task info for '{task}':")
            for style, prompt in task_info.items():
                print(f"   {style}: {prompt[:60]}...")
        
        print(f"\n✅ Loader test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_loader() 