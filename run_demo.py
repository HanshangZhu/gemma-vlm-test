#!/usr/bin/env python3
"""
TinyVLA Demo Launcher - Choose the right script for your needs

This script helps you select and run the appropriate TinyVLA demo based on your requirements.
"""

import os
import sys
import subprocess
import argparse

def print_demo_options():
    """Print available demo options"""
    print("🎮 TinyVLA Demo Options:")
    print("=" * 50)
    print()
    print("🏆 MAIN INFERENCE DEMO (Recommended):")
    print("   python tinyvla_inference_demo.py")
    print("   → Best performance, configurable diffusion steps, detailed metrics")
    print("   → Usage: --task pick-place-v3 --diffusion_steps 10 --fast")
    print()
    print("⚡ SPEED VARIANTS (inference_scripts/):")
    print("   python inference_scripts/realtime_metaworld_fast.py")
    print("   → Fastest performance, asynchronous inference")
    print()
    print("   python inference_scripts/realtime_metaworld_demo.py") 
    print("   → Action plotting and visual feedback")
    print()
    print("   python inference_scripts/realtime_metaworld_visual.py")
    print("   → Pure MuJoCo visualization")
    print()
    print("🔬 ANALYSIS TOOLS (analysis/):")
    print("   python analysis/diffusion_steps_comparison.py")
    print("   → Compare quality vs speed for different diffusion steps")
    print()
    print("   python analysis/reward_analysis.py") 
    print("   → Analyze MetaWorld reward structure")
    print()

def run_main_demo(args):
    """Run the main inference demo with provided arguments"""
    cmd = ["python", "tinyvla_inference_demo.py"]
    
    if args.task:
        cmd.extend(["--task", args.task])
    if args.diffusion_steps:
        cmd.extend(["--diffusion_steps", str(args.diffusion_steps)])
    if args.fast:
        cmd.append("--fast")
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print()
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="TinyVLA Demo Launcher")
    parser.add_argument("--list", action="store_true", 
                       help="List all available demo options")
    parser.add_argument("--task", type=str, default="pick-place-v3",
                       help="MetaWorld task name")
    parser.add_argument("--diffusion_steps", type=int, default=10,
                       help="Number of diffusion steps")
    parser.add_argument("--fast", action="store_true",
                       help="Enable fast mode")
    
    args = parser.parse_args()
    
    if args.list or len(sys.argv) == 1:
        print_demo_options()
        return
    
    run_main_demo(args)

if __name__ == "__main__":
    main() 