#!/usr/bin/env python3
"""Play GuanDan rounds with DanLM and print every play in human-readable form.

Demonstration script (requires the danzero binary extensions, i.e. macOS ARM64).

Examples:
    # DanLM (transformer) vs DanZero V1T (onnx), single round, seed 42
    PYTHONPATH=. python scripts/demo_round.py \
        --model-a ckpts/DanLM_v1/dansformer_v1_best_eval.pt \
        --model-b ckpts/DanZero_v3_rep_v1t/v3_rep_v1t_best_eval_001_int8.onnx \
        --seed 42

    # 3 complete games (level 2 -> A) vs a competition bot
    PYTHONPATH=. python scripts/demo_round.py \
        --model-a ckpts/DanLM_v1/dansformer_v1_best_eval.pt \
        --model-b bot:fin-njupt-guandan-ai \
        --games 3 --whole-game
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

from danzero.eval.agents import create_agent
from danzero.engine.actions import DIM_CARDS, DIM_PLAY_TYPE, play_type_of
from danzero.engine.cards import (
    BIG_JOKER,
    SMALL_JOKER,
    NUM_RANKS,
    TOTAL_CARD_TYPES,
    deal_hands,
    is_wild_card,
    level_rank_index,
    single_card_power,
)
from danzero.engine.game import (
    GuanDanRound,
    NUM_PLAYERS,
    compute_level_up,
)
from danzero.engine.tribute import (
    TributeRecord,
    perform_tribute,
    transfer_card,
    tribute_back_legal_cards,
    tribute_give_legal_cards,
)
from danzero.encoding.tokenizer import order_cards_in_play

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "S", "B"]
SUITS = ["H", "S", "C", "D"]
SEAT_NAMES = ["P0", "P1", "P2", "P3"]


def card_str(ci: int, level: int) -> str:
    if ci == SMALL_JOKER:
        return "小王"
    if ci == BIG_JOKER:
        return "大王"
    suit = SUITS[ci // NUM_RANKS]
    rank = RANKS[ci % NUM_RANKS]
    tag = ""
    if is_wild_card(ci, level):
        tag = "(百搭)"
    elif ci % NUM_RANKS == level_rank_index(level):
        tag = "(级)"
    return f"{suit}{rank}{tag}"


def play_str(play_80: np.ndarray, level: int) -> str:
    ptype = play_type_of(play_80)
    if ptype == "pass":
        return "不要"
    cards_ec = play_80[:DIM_CARDS]
    rank_oh = play_80[DIM_CARDS + DIM_PLAY_TYPE:]
    rank_idx = int(rank_oh.argmax())
    card_ints = order_cards_in_play(cards_ec, ptype, rank_idx, level_rank_index(level))
    return f"{ptype} [{' '.join(card_str(c, level) for c in card_ints)}]"


def hand_str(hand: np.ndarray, level: int) -> str:
    cards = []
    for ci in range(TOTAL_CARD_TYPES):
        for _ in range(int(hand[ci])):
            cards.append(card_str(ci, level))
    return " ".join(cards)


PTYPE_CN = {
    "single": "单张", "pair": "对子", "trips": "三张",
    "trio": "三连对", "fullhouse": "三带二", "plate": "钢板",
    "straight": "顺子", "straight_flush": "同花顺",
    "bomb": "炸弹", "joker_bomb": "王炸", "pass": "不要",
}


def describe_play(play_80: np.ndarray, level: int) -> str:
    ptype = play_type_of(play_80)
    return f"{PTYPE_CN.get(ptype, ptype)} {play_str(play_80, level)}"


def q_of(agent, obs, rnd, action_idx: int) -> float | None:
    try:
        q = agent.get_q_values(obs, rnd)
        return float(q[action_idx])
    except Exception:
        return None


def play_one_round(agent_a, agent_b, seed: int | None, verbose: bool = True) -> dict:
    """Play one single round (random level, 50% tribute) with verbose printing."""
    rng = np.random.default_rng(seed)

    level = int(rng.integers(2, 15))
    hands = deal_hands(seed=seed)
    has_tribute = bool(rng.integers(2))

    agents = {0: agent_a, 1: agent_b, 2: copy.copy(agent_a), 3: copy.copy(agent_b)}

    print(f"\n{'=' * 72}")
    print(f"新的一轮 | 级牌: {RANKS[level - 2]} | 种子: {seed}")
    print(f"{'=' * 72}")
    for p in range(NUM_PLAYERS):
        print(f"{SEAT_NAMES[p]} 手牌 ({int(hands[p].sum())} 张): {hand_str(hands[p], level)}")

    tribute_records: list[TributeRecord] = []
    anti_tribute_abs = None

    if has_tribute:
        finish_order = rng.permutation(NUM_PLAYERS).tolist()
        p1st, p2nd, p3rd, p4th = finish_order
        same_team = (p3rd - p4th) % 4 == 2
        print(f"\n-- 进贡阶段 (上一轮末位顺序: {[SEAT_NAMES[i] for i in finish_order]})")

        if same_team and hands[p4th][BIG_JOKER] + hands[p3rd][BIG_JOKER] >= 2:
            anti_tribute_abs = np.zeros((4, 54), dtype=np.float32)
            anti_tribute_abs[p4th, BIG_JOKER] = float(hands[p4th][BIG_JOKER])
            anti_tribute_abs[p3rd, BIG_JOKER] = float(hands[p3rd][BIG_JOKER])
            tribute_records, first_player = perform_tribute(hands, finish_order, level)
            print("   抗贡！双大王在手，无需进贡")
        elif not same_team and hands[p4th][BIG_JOKER] >= 2:
            anti_tribute_abs = np.zeros((4, 54), dtype=np.float32)
            anti_tribute_abs[p4th, BIG_JOKER] = float(hands[p4th][BIG_JOKER])
            tribute_records, first_player = perform_tribute(hands, finish_order, level)
            print(f"   {SEAT_NAMES[p4th]} 抗贡（双大王），直接首出")
        else:
            for pid, agent in agents.items():
                agent.reset(pid, level)

            givers = [p4th, p3rd] if same_team else [p4th]
            receivers = [None, None] if same_team else [p1st]

            for gi, giver in enumerate(givers):
                legal = tribute_give_legal_cards(hands[giver], level)
                if len(legal) == 1:
                    chosen = legal[0]
                else:
                    chosen = agents[giver].select_tribute_give(
                        hands[giver], legal, same_team, receivers[gi],
                    )
                rec = TributeRecord(giver=giver, receiver=-1, card=chosen)
                tribute_records.append(rec)

            if same_team:
                card_4th = tribute_records[0].card
                card_3rd = tribute_records[1].card
                power_4th = single_card_power(card_4th, level)
                power_3rd = single_card_power(card_3rd, level)
                if power_4th == power_3rd:
                    recv_4th = (p4th + 3) % 4
                    recv_3rd = (p3rd + 3) % 4
                elif power_4th > power_3rd:
                    recv_4th, recv_3rd = p1st, p2nd
                else:
                    recv_4th, recv_3rd = p2nd, p1st
                tribute_records[0].receiver = recv_4th
                tribute_records[1].receiver = recv_3rd
                receivers = [recv_4th, recv_3rd]
                transfer_card(hands, p4th, recv_4th, card_4th)
                transfer_card(hands, p3rd, recv_3rd, card_3rd)
                first_player = p4th if recv_4th == p1st else p3rd
            else:
                receiver = receivers[0]
                tribute_records[0].receiver = receiver
                card = tribute_records[0].card
                transfer_card(hands, givers[0], receiver, card)
                first_player = givers[0]

            give_records = list(tribute_records)
            for bi, returner in enumerate(receivers):
                back_to = givers[bi]
                legal = tribute_back_legal_cards(hands[returner], level)
                if len(legal) == 1:
                    chosen = legal[0]
                else:
                    chosen = agents[returner].select_tribute_back(
                        hands[returner], legal, back_to, give_records,
                    )
                rec = TributeRecord(giver=returner, receiver=back_to, card=chosen)
                tribute_records.append(rec)
                transfer_card(hands, returner, back_to, chosen)

            for rec in tribute_records:
                src = SEAT_NAMES[rec.giver]
                dst = SEAT_NAMES[rec.receiver] if rec.receiver >= 0 else "?"
                print(f"   进贡: {src} -> {dst}  {card_str(rec.card, level)}")
    else:
        first_player = int(rng.integers(NUM_PLAYERS))
        print("\n-- 无进贡")

    for pid, agent in agents.items():
        agent.reset(pid, level)
    for agent in agents.values():
        agent.notify_tribute(tribute_records, anti_tribute_abs=anti_tribute_abs)

    rnd = GuanDanRound(
        level=level, hands=hands, first_player=first_player,
        team_levels=(level, level),
    )
    for pid, agent in agents.items():
        agent.notify_start(rnd.state.hands[pid].copy())

    print(f"\n-- 首出: {SEAT_NAMES[first_player]}")
    print(f"\n{'-' * 72}")

    obs = rnd.get_observation()
    trick_no = 1
    steps = 0
    while not rnd.done and steps < 2000:
        p = obs.player
        agent = agents[p]
        action_idx = agent.select_play(obs, rnd)
        play = obs.legal_plays[action_idx].copy()
        q = q_of(agent, obs, rnd, action_idx)
        q_txt = f"  Q={q:+.3f}" if q is not None else ""
        lead = " [领出]" if obs.is_leading else ""
        print(f"  第{trick_no:3d}手 {SEAT_NAMES[p]}{lead}: {describe_play(play, level)} "
              f"(剩{int(rnd.state.hands[p].sum())}张){q_txt}")
        obs = rnd.step(action_idx)
        steps += 1
        new_trick = obs is not None and obs.is_leading
        for agent2 in agents.values():
            agent2.observe_action(p, play, new_trick)
        if new_trick:
            trick_no += 1
        if obs is None:
            break

    fo = rnd.finish_order
    team_a_win = fo[0] % 2 == 0  # team 0 = seats 0,2 (agent A)
    print(f"{'-' * 72}")
    print(f"  结束! 完成 order: {[SEAT_NAMES[i] for i in fo]}")
    print(f"  获胜队伍: {'A队 (P0+P2, DanLM)' if team_a_win else 'B队 (P1+P3, 对手)'}")
    return {"winner_team": 0 if team_a_win else 1, "rounds": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="DanLM verbose round demo")
    parser.add_argument("--model-a", type=str, required=True)
    parser.add_argument("--model-b", type=str, default="random")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("加载模型...")
    agent_a = create_agent(args.model_a, args.device)
    agent_b = create_agent(args.model_b, args.device)
    print(f"  A = {args.model_a}")
    print(f"  B = {args.model_b}")

    wins = 0
    for g in range(args.games):
        seed = args.seed + g * 100_000 if args.seed is not None else None
        result = play_one_round(agent_a, agent_b, seed)
        if result["winner_team"] == 0:
            wins += 1
    print(f"\n{'=' * 72}")
    print(f"总计: A队(DanLM) 胜 {wins}/{args.games}")


if __name__ == "__main__":
    main()
