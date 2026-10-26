"""Print the system and first user prompt RPent gives the planner under --knowledge-profile proven.

usage: python rocm/render_prompt.py <RPent checkout with the patch> <suite> <task> <seed>
"""

import sys
from pathlib import Path

root, suite, task, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, str(root))
from robots.libero.prompt_bundle import system_prompt, user_prompt  # noqa: E402
from rpent.context.prompt_utils import format_prompt  # noqa: E402

short = suite.replace("libero_", "")
v = {"suite": suite, "task": task, "seed": seed, "recipe_tag": f"{short}_t{task}_s{seed}", "mode": "eval",
     "memory_profile": "hf", "knowledge_profile": "proven", "memory_dir": "<memory>",
     "reference_tag": f"{short}_t{task}_s0", "memory_inbox": "<inbox>", "session_number": 1, "session_max": 1,
     "output_dir": "<episode_dir>"}
print("=== SYSTEM ===")
print(format_prompt(system_prompt(v), variables=v))
print("=== USER ===")
print(format_prompt(user_prompt(v), variables=v))
