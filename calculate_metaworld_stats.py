#!/usr/bin/env python3
"""
Calculates normalization statistics from the short-MetaWorld dataset
and saves them to a pickle file.
"""
import pickle
import numpy as np
import os

def calculate_stats(data_path, output_path):
    print(f"Loading data from: {data_path}")
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # Extract state and action data using the correct keys
    all_states_raw = data['state']    # Shape: (100, 20, 39)
    all_actions_raw = data['actions'] # Shape: (100, 20, 4)

    # The model expects a 7-dimensional state. We will take the first 7 elements.
    # This is an assumption that may need to be revisited.
    # The important part is to be consistent.
    all_qpos = all_states_raw[:, :, :7]
    all_actions = all_actions_raw

    # Reshape the data from (num_traj, traj_len, dim) to (num_traj * traj_len, dim)
    num_trajectories = all_qpos.shape[0]
    all_qpos_np = all_qpos.reshape(-1, all_qpos.shape[-1])
    all_actions_np = all_actions.reshape(-1, all_actions.shape[-1])

    print(f"Collected {all_qpos_np.shape[0]} states from {num_trajectories} trajectories.")
    print(f"  State shape for stats: {all_qpos_np.shape}")
    print(f"  Action shape for stats: {all_actions_np.shape}")

    # Calculate statistics
    stats = {
        'qpos_mean': all_qpos_np.mean(axis=0),
        'qpos_std': all_qpos_np.std(axis=0),
        'action_mean': all_actions_np.mean(axis=0),
        'action_std': all_actions_np.std(axis=0),
        'action_min': all_actions_np.min(axis=0),
        'action_max': all_actions_np.max(axis=0),
    }
    
    # Ensure std is not zero to avoid division by zero errors
    stats['qpos_std'][stats['qpos_std'] == 0] = 1e-6
    stats['action_std'][stats['action_std'] == 0] = 1e-6


    print("\n--- Calculated Statistics ---")
    for key, value in stats.items():
        print(f"  {key}: shape={value.shape}")
    print("--------------------------")

    # Save to output file
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"\n✅ Statistics saved to: {output_path}")


if __name__ == "__main__":
    data_file = "datasets/short-MetaWorld/short-MetaWorld/r3m-processed/r3m_MT10_20/pick-place-v2.pkl"
    output_file = "metaworld_stats.pkl"
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found at {data_file}")
    else:
        calculate_stats(data_file, output_file) 