#!/usr/bin/env python3
"""Diagnose why bot agents silently fail in evaluate() on the runner."""

import sys
import traceback

sys.path.insert(0, ".")

print("== 1. create_agent('bot:fin-njupt-guandan-ai') ==")
try:
    from danzero.eval.agents import create_agent

    agent = create_agent("bot:fin-njupt-guandan-ai", "cpu")
    print("OK:", type(agent))
except Exception:
    traceback.print_exc()

print()
print("== 2. play 1 game vs the bot ==")
try:
    from danzero.eval.agents import create_agent
    from danzero.eval.evaluator import evaluate

    a = create_agent("ckpts/DanLM_v1/dansformer_v1_best_eval.pt", "cpu")
    b = create_agent("bot:fin-njupt-guandan-ai", "cpu")
    res = evaluate(a, b, num_games=1, seed=7, log_interval=0)
    print("result:", res)
except Exception:
    traceback.print_exc()

print()
print("== 3. import bot module directly ==")
try:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diag_bot_action", "baselines/fin-njupt-guandan-ai/action.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("bot action.py import OK:", [n for n in dir(mod) if not n.startswith("_")][:10])
except Exception:
    traceback.print_exc()
