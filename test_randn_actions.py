#!/usr/bin/env python3
"""
🎲 Random Action Visual Test

This test uses completely random actions to verify:
1. Robot moves visually when actions are applied
2. Visual rendering updates properly with unpredictable movement
3. Movement tracking works with random behavior
"""

import os
import time
import numpy as np
import metaworld

def test_random_actions():
    """Test with random actions to verify visual movement"""
    
    print("🎲 Random Action Visual Test")
    print("=" * 50)
    print("💡 This test uses random actions to verify robot movement is visible")
    
    # Enable visual rendering
    if 'DISPLAY' in os.environ:
        os.environ['MUJOCO_GL'] = 'glfw'
        print("🖼️ Using GLFW for visual rendering - GUI window will open")
    else:
        os.environ['MUJOCO_GL'] = 'egl'
        print("🖼️ Using EGL for headless rendering")
    
    # Create environment
    print("\n1️⃣ Creating MetaWorld environment...")
    ml1 = metaworld.ML1('pick-place-v2', seed=42)
    env = ml1.train_classes['pick-place-v2']()
    task = list(ml1.train_tasks)[0]
    env.set_task(task)
    
    # Reset and get initial state
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    initial_pos = obs[:3]
    print(f"✅ Environment ready")
    print(f"🤖 Initial robot position: {initial_pos}")
    print(f"🎯 Action space: {env.action_space}")
    print(f"📊 Action bounds: low={env.action_space.low}, high={env.action_space.high}")
    
    # Initial render with pause
    print("\n2️⃣ Rendering initial state...")
    try:
        env.render()
        print("✅ MuJoCo window should be open now!")
        print("🖼️ You should see the red robot arm in starting position")
        time.sleep(2.0)  # 2 second pause to see initial state
    except Exception as e:
        print(f"❌ Initial render failed: {e}")
        return
    
    print("\n3️⃣ Starting random action test...")
    print("🎲 Taking 30 random actions with visual updates")
    print("💡 Each action will be held for 100ms - watch for robot movement!")
    
    # Track movement statistics
    max_movement = 0.0
    total_movement = 0.0
    positions = [initial_pos.copy()]
    
    for step in range(30):
        # Generate random action
        action = env.action_space.sample()
        
        print(f"\n   🎯 Step {step+1:2d}/30:")
        print(f"      🎲 Random action: {action}")
        
        # Take action
        obs, reward, done, info = env.step(action)
        
        # Track position
        current_pos = obs[:3]
        positions.append(current_pos.copy())
        
        # Calculate movement from previous position
        step_movement = np.linalg.norm(current_pos - positions[-2])
        total_movement_from_start = np.linalg.norm(current_pos - initial_pos)
        
        if total_movement_from_start > max_movement:
            max_movement = total_movement_from_start
        
        total_movement += step_movement
        
        print(f"      📊 Position: {current_pos}")
        print(f"      📏 Step movement: {step_movement:.4f} units")
        print(f"      📐 Total from start: {total_movement_from_start:.4f} units")
        print(f"      🏆 Reward: {reward:.3f}")
        
        # Visual update with timing
        render_start = time.time()
        try:
            env.render()
            # Additional render calls to ensure visual update
            if hasattr(env, 'viewer') and env.viewer is not None:
                if hasattr(env.viewer, 'render'):
                    env.viewer.render()
        except Exception as e:
            print(f"      ⚠️ Render failed: {e}")
        
        render_time = time.time() - render_start
        print(f"      ✅ Visual frame updated ({render_time*1000:.1f}ms)")
        
        # Pause to make movement visible
        time.sleep(0.1)  # 100ms delay
        
        # Show progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"\n   📈 Progress Update ({step+1}/30):")
            print(f"      🏔️ Max movement from start: {max_movement:.4f} units")
            print(f"      🛤️ Total movement distance: {total_movement:.4f} units")
            print(f"      📍 Current position: {current_pos}")
    
    # Final statistics
    print(f"\n✅ Random action test completed!")
    print(f"\n📊 MOVEMENT STATISTICS:")
    print(f"   🏁 Total steps: 30")
    print(f"   🏔️ Maximum distance from start: {max_movement:.4f} units")
    print(f"   🛤️ Total movement distance: {total_movement:.4f} units")
    print(f"   📍 Starting position: {initial_pos}")
    print(f"   📍 Final position: {current_pos}")
    print(f"   📏 Net displacement: {np.linalg.norm(current_pos - initial_pos):.4f} units")
    
    # Movement analysis
    if max_movement > 0.01:
        print(f"\n✅ GOOD: Robot showed significant movement (>{0.01:.3f} units)")
        print(f"🖼️ Visual movement should have been clearly visible!")
    elif max_movement > 0.001:
        print(f"\n⚠️ MODERATE: Robot showed some movement ({max_movement:.4f} units)")
        print(f"🖼️ Movement might be subtle - try zooming in the MuJoCo window")
    else:
        print(f"\n❌ CONCERN: Very little movement detected ({max_movement:.4f} units)")
        print(f"🔧 There might be an issue with action execution or constraints")
    
    # Final pause
    print(f"\n⏸️ Test ending in 3 seconds...")
    time.sleep(3.0)
    
    # Clean up
    try:
        env.close()
    except:
        pass

if __name__ == "__main__":
    test_random_actions()