#!/usr/bin/env python3
"""Diag v2: does adding the bot dir to sys.path fix the silent failure?"""

import sys
import traceback

sys.path.insert(0, ".")

print("== A. without extra sys.path ==")
try:
    from danzero.eval.agents import create_agent
    from danzero.eval.evaluator import evaluate

    a = create_agent("ckpts/DanLM_v1/dansformer_v1_best_eval.pt", "cpu")
    b = create_agent("bot:fin-njupt-guandan-ai", "cpu")
    print("bot agent type:", type(b))
    res = evaluate(a, b, num_games=1, seed=7, log_interval=0)
    print("result:", res)
except Exception:
    traceback.print_exc()

print()
print("== B. with bot dir in sys.path ==")
try:
    sys.path.insert(0, "baselines/fin-njupt-guandan-ai")
    from danzero.eval.agents import create_agent as create_agent2
    from danzero.eval.evaluator import evaluate as evaluate2

    a2 = create_agent2("ckpts/DanLM_v1/dansformer_v1_best_eval.pt", "cpu")
    b2 = create_agent2("bot:fin-njupt-guandan-ai", "cpu")
    print("bot agent type:", type(b2))
    res2 = evaluate2(a2, b2, num_games=1, seed=7, log_interval=0)
    print("result:", res2)
except Exception:
    traceback.print_exc()
