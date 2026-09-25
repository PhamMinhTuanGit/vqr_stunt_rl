"""Summarize yaw FSM wheel-contact traces written by rsl_rl/play.py."""

from __future__ import annotations

import argparse
import csv
from collections import Counter


SUPPORT_WHEELS = {2: ("FL", "HR"), 4: ("FR", "HL")}


def analyze(path: str, threshold: float, airborne_clearance: float, near_ground: float) -> None:
    counts: dict[str, Counter[str]] = {"POS": Counter(), "NEG": Counter()}
    direction_envs: dict[str, set[int]] = {"POS": set(), "NEG": set()}
    state_counts: Counter[int] = Counter()
    command_counts: Counter[str] = Counter()
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            state = int(row["fsm_state"])
            state_counts[state] += 1
            command = float(row["yaw_command"])
            if command < -0.10:
                command_counts["negative_enter"] += 1
                if state == 5:
                    command_counts["negative_enter_in_return"] += 1
            elif command > 0.10:
                command_counts["positive_enter"] += 1
            if state not in SUPPORT_WHEELS:
                continue
            direction = "POS" if state == 2 else "NEG"
            direction_envs[direction].add(int(row["env_id"]))
            result = counts[direction]
            result["yaw_steps"] += 1
            wheels = SUPPORT_WHEELS[state]
            norm_flags = []
            fz_flags = []
            for wheel in wheels:
                clearance = float(row[f"{wheel}_clearance_m"])
                fz = float(row[f"{wheel}_fz_n"])
                norm = float(row[f"{wheel}_force_norm_n"])
                norm_flag = norm > threshold
                fz_flag = fz > threshold
                norm_flags.append(norm_flag)
                fz_flags.append(fz_flag)
                result["wheel_samples"] += 1
                result["norm_contact"] += norm_flag
                result["fz_contact"] += fz_flag
                result["norm_without_fz"] += norm_flag and not fz_flag
                result["airborne_zero_force"] += clearance > airborne_clearance and norm < 0.2 * threshold
                result["near_ground_threshold_band"] += (
                    abs(clearance) <= near_ground and 0.5 * threshold <= fz <= 1.5 * threshold
                )
                history_norms = [
                    float(value)
                    for key, value in row.items()
                    if key.startswith(f"{wheel}_history_") and key.endswith("_norm_n") and value != ""
                ]
                result["history_current_miss"] += (
                    not norm_flag and any(value > threshold for value in history_norms)
                )
            norm_pair = all(norm_flags)
            fz_pair = all(fz_flags)
            result["norm_pair"] += norm_pair
            result["fz_pair"] += fz_pair
            result["norm_pair_without_fz_pair"] += norm_pair and not fz_pair
            metric = row.get("metric_support_gate", "")
            if metric != "":
                result["metric_samples"] += 1
                result["metric_mismatch"] += (float(metric) > 0.5) != norm_pair

    print(f"FSM state samples: {dict(sorted(state_counts.items()))}")
    print(f"Commands beyond entry threshold: + {command_counts['positive_enter']}, - {command_counts['negative_enter']} ({command_counts['negative_enter_in_return']} while RETURN_TO_4)")
    for direction, result in counts.items():
        steps = result["yaw_steps"]
        wheels = result["wheel_samples"]
        print(f"{direction}: {steps} YAW steps from {len(direction_envs[direction])} environments")
        if not steps:
            continue
        print(f"  both support wheels: ||F||>{threshold:g} N {result['norm_pair'] / steps:.1%}; Fz>{threshold:g} N {result['fz_pair'] / steps:.1%}")
        print(f"  norm pair passes but Fz pair fails: {result['norm_pair_without_fz_pair']} steps")
        print(f"  individual wheel samples: norm without Fz {result['norm_without_fz']} / {wheels}")
        print(f"  clearance>{airborne_clearance:g} m and ||F||<{0.2 * threshold:g} N: {result['airborne_zero_force']} / {wheels}")
        print(f"  |clearance|<={near_ground:g} m and Fz near threshold: {result['near_ground_threshold_band']} / {wheels}")
        print(f"  current ||F|| fails while a history sample passes: {result['history_current_miss']} / {wheels}")
        print(f"  recorded metric differs from recalculated ||F|| pair: {result['metric_mismatch']} / {result['metric_samples']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_csv")
    parser.add_argument("--threshold", type=float, default=1.0, help="Contact-force threshold in N.")
    parser.add_argument("--airborne-clearance", type=float, default=0.02, help="Airborne wheel clearance in m.")
    parser.add_argument("--near-ground", type=float, default=0.01, help="Near-ground clearance band in m.")
    args = parser.parse_args()
    analyze(args.trace_csv, args.threshold, args.airborne_clearance, args.near_ground)
