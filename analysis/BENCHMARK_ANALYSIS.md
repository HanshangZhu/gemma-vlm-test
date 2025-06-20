# TinyVLA Benchmark Analysis: The Training-Evaluation Gap

## 1. Executive Summary

This document analyzes the results of a systematic benchmark of four TinyVLA diffusion policy checkpoints, trained for 10, 20, 30, and 40 epochs. The goal was to diagnose a previously observed "reward hacking" behavior where the model achieved high reward but 0% task success.

**The key finding is that the model suffers from a severe training-to-evaluation distribution gap.** Despite achieving excellent (low) training loss, the policy completely fails to generalize to a live environment, resulting in a **0.0% success rate across all checkpoints.** Our initial hypothesis of simple overfitting was incorrect; the problem is more fundamental and lies within the training data itself.

## 2. Benchmark Results

The following table summarizes the performance of each checkpoint on a 20-episode `pick-place-v2` benchmark.

| Checkpoint | Avg. Training Loss | Avg. Evaluation Reward | **Success Rate** | Reached Object | Grasped Object |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Epoch 10** | ~0.26 | 0.935 | **0.0%** | 0.0% | 0.0% |
| **Epoch 20** | ~0.16 | 2.743 | **0.0%** | 0.0% | 0.0% |
| **Epoch 30** | ~0.10 | 1.187 | **0.0%** | 0.0% | 0.0% |
| **Epoch 40** | ~0.09 | 0.863 | **0.0%** | 0.0% | 0.0% |

### Key Observations:

- **Zero Success:** No checkpoint was able to complete the pick-place task a single time.
- **Failure to Generalize:** The models failed at the most basic stage of the task (reaching the object), indicating that the learned policy is not robust to minor variations between the training data and the live environment.
- **Misleading Rewards:** The fluctuating average reward is not correlated with success and is likely a product of the policy executing arbitrary, non-productive movements.
- **Excellent Loss, Poor Performance:** The extremely low training loss proves the model is effectively "memorizing" the training trajectories but is not learning the underlying skills required for manipulation.

## 3. Diagnosis: Training-Evaluation Distribution Gap

The data provides a clear diagnosis: the training dataset, while internally consistent, is not diverse enough to produce a policy that can generalize. The model learns a "brittle" solution that is hyper-specific to the training examples and breaks down when faced with even small, real-world perturbations.

This is a common problem in imitation learning and robotics. The model is not overfitting in the traditional sense (where performance on a validation set degrades); rather, it is "overfitting" to the entire training distribution, which itself is too narrow.

## 4. Recommended Next Steps

The path forward requires improving the quality and diversity of our training data. The goal is to "bridge the gap" between the training environment and the real world.

### Step 1: Analyze the Existing Dataset

Before augmenting or collecting new data, we must understand the limitations of our current dataset. I will create a script to:
- **Visualize Trajectories:** Plot the end-effector paths from the `.pkl` files to see how similar they are.
- **Analyze Action Distributions:** Check the range and frequency of the recorded actions (position, rotation, gripper state).
- **Inspect Image Data:** Review the camera images to check for lack of variety in lighting, object position, etc.

### Step 2: Implement Data Augmentation

Based on the analysis, we will implement data augmentation techniques in the training pipeline. This is the most direct way to improve policy robustness without collecting new data. Potential augmentations include:
- **Visual Augmentations:** Random cropping, color jitter, brightness/contrast shifts.
- **State/Action Augmentations:** Adding small amounts of Gaussian noise to robot states and action vectors.

By tackling the data problem head-on, we can move from a model that simply memorizes to one that truly learns and generalizes.