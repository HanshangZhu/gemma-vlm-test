1. Overview
Short-MetaWorld is a dataset rendered from modified environment of Meta-World [1], which contains Multi-Task10(MT10) and Meta-Learning10(ML10) in total 20 tasks with 100 successful trajectories for each task. Each trajectory is padded to 20 steps.

2. File Structure
This directory contains 3 sub-directories.

└── short-MetaWorld
    ├── task_description.py           # language instructions for each task
    ├── img_only                      # rendered visual inputs for all tasks
    │   ├── button-press-topdown-v2   # task name
    │   │     ├── 0                   # trajectory id     
    │   │     │   ├── 0.jpg           # step 0 observation (224*224)
    │   │     │	  ├── 1.jpg           # step 1 observation
    │   │     │   └── ...
    │   │     ├── 1
    │   │     └── ...
    │   ├── door-open-v2
    │   └── ...
    ├── unprocessed     	   # trajectory actions with visual observations (256*256)
    │   ├── unprocessed_MT10_20    # task name
    │   │   ├── data.pkl           # a file contains all 10 tasks
    │   │   ├── door-open-v2.pkl   # door-open task file
    │   │   └── ...
    │   └── unprocessed_ML10_20
    └── r3m-processed              # trajectory actions with visual obs processed by R3M [2]

3. Contact
If you have any questions, please contact liangzx@connect.hku.hk

[1] Yu, Tianhe, et al. "Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning." Conference on robot learning. PMLR, 2020.
[2] Nair, Suraj, et al. "R3M: A Universal Visual Representation for Robot Manipulation." Conference on Robot Learning. PMLR, 2023.


