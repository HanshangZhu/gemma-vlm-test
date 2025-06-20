# TinyVLA Real-Time GUI User Guide

## Overview

The TinyVLA Real-Time GUI provides an interactive interface for testing our trained diffusion policy with live visualization. You can experiment with different prompts, tasks, and settings while watching the robot perform in real-time.

## Features

### 🎮 **Interactive Controls**
- **Task Selection**: Choose from 6 different MetaWorld tasks
- **Prompt Engineering**: Test different instruction styles
- **Episode Settings**: Adjust maximum steps (50-200)
- **Real-Time Control**: Start, stop, and reset episodes

### 📺 **Live Visualization**
- **Robot Camera View**: See what the robot sees in real-time
- **Reward Plotting**: Live graph of cumulative reward over time
- **Status Monitoring**: Track episode progress and performance

### 🚀 **Quick Prompt Testing**
- Pre-configured prompt buttons for instant testing
- Compare simple vs detailed instructions
- Optimize prompts for maximum performance

## Getting Started

### 1. **Launch the GUI**
```bash
# Method 1: Direct launch
python inference_scripts/tinyvla_realtime_gui.py

# Method 2: Using launcher
python launch_gui.py
```

### 2. **GUI Layout**

```
┌─────────────────────────────────────────────────────────────┐
│                    TinyVLA Real-Time GUI                    │
├──────────────┬──────────────────────────────────────────────┤
│   Controls   │              Live Visualization             │
│              │                                              │
│ Task: [▼]    │  ┌────────────────────────────────────────┐  │
│ Prompt: [...] │  │        Robot Camera View               │  │
│              │  │                                        │  │
│ Quick Prompts│  │         [Live Video Feed]              │  │
│ [Simple]     │  │                                        │  │
│ [Detailed]   │  └────────────────────────────────────────┘  │
│ [Reach]      │                                              │
│ [Button]     │  ┌────────────────────────────────────────┐  │
│ [Minimal]    │  │         Reward Over Time               │  │
│              │  │                                        │  │
│ Max Steps: 100│  │      [Live Reward Graph]               │  │
│              │  │                                        │  │
│ [Start Episode]│  └────────────────────────────────────────┘  │
│ [Stop]       │                                              │
│ [Reset]      │                                              │
│              │                                              │
│ Status: Ready│                                              │
│ Episode: 0   │                                              │
│ Step: 0      │                                              │
│ Reward: 0.000│                                              │
└──────────────┴──────────────────────────────────────────────┘
```

## Using the GUI

### 🎯 **Basic Workflow**

1. **Select Task**: Choose from dropdown (pick-place-v2, reach-v2, etc.)
2. **Enter Prompt**: Type instruction or use quick prompt buttons
3. **Set Max Steps**: Adjust episode length (default: 100)
4. **Start Episode**: Click "Start Episode" to begin
5. **Watch Live**: Monitor robot camera and reward graph
6. **Analyze Results**: Check final reward and success status

### 📝 **Prompt Engineering Workflow**

1. **Start with Simple**: Use "Simple" quick prompt button
2. **Run Episode**: Observe performance and final reward
3. **Try Variations**: Test "Detailed", "Minimal" prompts
4. **Compare Results**: Note which prompts work best
5. **Optimize**: Create custom prompts based on findings

### 🔧 **Advanced Usage**

#### **Task-Specific Optimization**
```
Pick-Place Tasks:
1. Start with "Pick up the object and place it at the target location"
2. Try "pick and place" for comparison
3. Avoid detailed color descriptions

Reach Tasks:
1. Use "Reach to the target position"
2. Compare with "Move to the target"
3. Avoid single-word prompts

Button Tasks:
1. Start with "Press the button"
2. Try "Push the button"
3. Avoid technical descriptions
```

#### **Episode Length Optimization**
```
Short Episodes (50-75 steps):
- Good for reach tasks
- Quick testing of prompts
- Fast iteration

Medium Episodes (100-125 steps):
- Standard for most tasks
- Good balance of time/completion

Long Episodes (150-200 steps):
- Complex manipulation tasks
- Maximum completion chance
- Detailed analysis
```

## GUI Controls Reference

### 🎮 **Control Panel**

| Control | Function | Notes |
|---------|----------|-------|
| **Task Dropdown** | Select MetaWorld task | Automatically maps v2→v3 |
| **Prompt Entry** | Enter custom instruction | 4-8 words optimal |
| **Quick Prompts** | Pre-configured buttons | Based on our research |
| **Max Steps** | Episode length | 50-200 range |
| **Start Episode** | Begin new episode | Disabled during run |
| **Stop** | Halt current episode | Emergency stop |
| **Reset** | Clear all data | Reset counters |

### 📊 **Status Display**

| Field | Shows | Example |
|-------|-------|---------|
| **Status** | Current state | "Running episode...", "SUCCESS!" |
| **Episode** | Episode counter | 1, 2, 3... |
| **Step** | Current step | 0-200 |
| **Reward** | Cumulative reward | 0.000-100.000 |

### 📺 **Visualization**

| Panel | Content | Updates |
|-------|---------|---------|
| **Robot Camera** | Live RGB feed | Real-time (10 FPS) |
| **Reward Graph** | Cumulative reward | Every step |

## Performance Interpretation

### 🏆 **Success Indicators**

#### **Pick-Place Tasks**
- **Excellent**: Reward > 1.0, smooth object manipulation
- **Good**: Reward 0.5-1.0, reaches object consistently
- **Poor**: Reward < 0.5, erratic movements

#### **Reach Tasks**
- **Excellent**: Reward > 80, precise target reaching
- **Good**: Reward 50-80, generally correct direction
- **Poor**: Reward < 50, random movements

#### **Button Tasks**
- **Excellent**: Reward > 70, clear button approach
- **Good**: Reward 40-70, moves toward button
- **Poor**: Reward < 40, no clear intent

### 📈 **Reward Graph Analysis**

```
Healthy Patterns:
- Steady increase over time
- Smooth curve without oscillations
- Final plateau at high value

Warning Signs:
- Flat line (no progress)
- Oscillating values (unstable)
- Sudden drops (failure)
```

## Troubleshooting

### ❌ **Common Issues**

#### **GUI Won't Start**
```bash
# Check conda environment
conda activate tinyvla

# Check dependencies
pip install tkinter matplotlib

# Check model path
ls checkpoints/TinyVLA-droid_diffusion_metaworld/
```

#### **Model Loading Failed**
```bash
# Verify checkpoint exists
ls -la checkpoints/TinyVLA-droid_diffusion_metaworld/diff_head_FIXED_epoch_40.pth

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### **Environment Setup Failed**
```bash
# Check MetaWorld installation
python -c "import metaworld; print('MetaWorld OK')"

# Check task name
# Use v2 names (pick-place-v2), GUI handles v3 mapping
```

#### **Poor Performance**
```
1. Try different prompts (use quick buttons)
2. Increase max steps for complex tasks
3. Check if model loaded correctly
4. Verify task selection matches prompt
```

### 🔧 **Performance Tips**

1. **Start Simple**: Use quick prompt buttons first
2. **Iterate Fast**: Use shorter episodes for prompt testing
3. **Compare Systematically**: Test one variable at a time
4. **Watch Patterns**: Look for consistent behaviors
5. **Document Results**: Note which prompts work best

## Best Practices

### 🎯 **Prompt Engineering**
1. **Start with Quick Prompts**: Use pre-configured buttons
2. **Keep It Simple**: 4-8 words work best
3. **Use Action Verbs**: pick, place, reach, press
4. **Avoid Details**: No colors, adverbs, technical terms
5. **Test Systematically**: One change at a time

### 📊 **Performance Testing**
1. **Multiple Episodes**: Run 3-5 episodes per prompt
2. **Consistent Settings**: Same max steps for comparison
3. **Document Results**: Track best prompts for each task
4. **Look for Patterns**: Consistent behaviors across runs

### 🎮 **GUI Usage**
1. **Monitor Live**: Watch both camera and reward graph
2. **Stop Early**: If clearly failing, stop and try new prompt
3. **Reset Between**: Clear data between major changes
4. **Save Findings**: Note successful prompt/task combinations

## Advanced Features

### 🔬 **Research Mode**
- Use GUI for systematic prompt engineering research
- Test hypothesis about language instruction effectiveness
- Compare performance across different tasks
- Generate data for analysis

### 📹 **Video Recording**
- Robot camera view shows live behavior
- Can screenshot interesting moments
- Reward graphs show performance patterns
- Useful for presentations and analysis

### 🎛️ **Parameter Tuning**
- Adjust max steps for different task complexity
- Test edge cases with unusual prompts
- Explore model limitations and capabilities

---

**The TinyVLA Real-Time GUI makes it easy to explore the capabilities of your trained diffusion policy and optimize performance through interactive prompt engineering!** 🚀 