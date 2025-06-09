#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Environment Setup ---
# This section ensures that the correct Conda environment and Python paths are set.

echo ">>> Activating Conda environment 'tinyvla'..."
# Source Conda's shell functions to make 'conda' command available.
eval "$(conda shell.bash hook)"
conda activate tinyvla

# Define the base path for your project to make the script more readable.
PROJECT_BASE_PATH="/home/hz/gemma-vlm-test/TinyVLA"
echo ">>> Project base path set to: $PROJECT_BASE_PATH"

# Add the project's root and the llava-pythia subdirectory to PYTHONPATH.
# This allows Python to find the custom modules like `llava_pythia` and `policy_heads`.
export PYTHONPATH="${PYTHONPATH}:${PROJECT_BASE_PATH}:${PROJECT_BASE_PATH}/llava-pythia"
echo ">>> PYTHONPATH configured."

# --- Dependency Installation ---
# This ensures that the 'policy_heads' package is installed in editable mode.
# The '-e' flag is useful for development, as changes to the source code
# are immediately reflected without needing to reinstall.

echo ">>> Ensuring 'policy_heads' dependency is installed..."
pip install --quiet -e "${PROJECT_BASE_PATH}/policy_heads/"
echo ">>> Dependencies are up to date."

# --- Execute the Python Script ---
# This command runs the main Python inference script.
# The `"$@"` passes along any command-line arguments you might provide to this shell script,
# offering future flexibility.

echo ">>> Running the TinyVLA inference script..."
echo "--------------------------------------------------"
# Use python3 to be explicit, and reference the script in the parent directory.
python3 unified_tinyvla.py "$@"
echo "--------------------------------------------------"
echo ">>> Script execution finished."
