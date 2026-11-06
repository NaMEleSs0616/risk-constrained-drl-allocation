#!/usr/bin/env python
"""Evaluation-only robustness of the next-open ETF study (no retraining).

Every trained fold model of every (variant, seed) chain and every baseline is re-run on
its out-of-sample window under perturbed execution assumptions:

* ``base``            - the configuration used in the study (reproduces the main table);
* ``cost_x0``         - frictionless (no fees, spread, impact or participation cap);
* ``cost_x0.5/x2/x3`` - all friction coefficients scaled;
* ``spread_only``     - fees and half-spread only (no square-root impact);
* ``impact_x3``       - square-root impact coefficients tripled, fees and spread unchanged;
* ``exec_next_close`` - one more session of delay: fill at the close of t+1;
* ``exec_close``      - the legacy same-close fill (NOT tradable; shows how much the
                        convention flatters the same policies);
* ``init_1N``         - every fold starts from an inherited 1/N book instead of cash.

The policies were trained under ``base``; the perturbations test whether conclusions
depend on the exact cost and timing assumptions, not what a policy trained under
other assumptions would do.

Also writes ``reward_clipping.csv``: how often the reward clip binds, (i) out of sample
and (ii) when each fold's final policy is replayed on its own training window in
252-day episodes (the training episode length), for each variant.

Usage:
    python scripts/sensitivity_analysis.py --results outputs_real_nextopen
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.run_experiments import _init_worker, build_variants, slugify, stitch  # noqa: E402
from src.backtest import load_policy, make_eval_env, run_episode, walk_forward_splits  # noqa: E402
from src.baselines import build_baselines  # noqa: E402
from src.environment import make_portfolio_env  # noqa: E402
from src.inference import METRICS, minimum_detectable_effect, two_level_bootstrap  # noqa: E402
from src.train import first_valid_index, prepare_data  # noqa: E402
from src.utils import configure_logging, get_logger, load_config  # noqa: E402

logger = get_logger("sensitivity")
EW = "Equal Weight (1/N)"
UNTRAINED = "Untrained policy (1/(N+1) incl. cash)"
COST_KEYS = ("commission_bps", "slippage_bps", "temp_impact_coef", "perm_impact_coef")


def scenarios(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Environment overrides (and an optional inherited book) for every scenario."""
    fr = dict(cfg["environment"]["friction"])

    def scaled(k: float, keys: Sequence[str] = COST_KEYS, **extra: Any) -> dict[str, Any]:
        return {"friction": {**fr, **{x: k * float(fr[x]) for x in keys}, **extra}}

    return {
        "base": {},
        "cost_x0": scaled(0.0, max_participation=None),
        "cost_x0.5": scaled(0.5),
        "cost_x2": scaled(2.0),
        "cost_x3": scaled(3.0),
        "spread_only": scaled(0.0, ("temp_impact_coef", "perm_impact_coef")),
        "impact_x3": scaled(3.0, ("temp_impact_coef", "perm_impact_coef")),
        "exec_next_close": {"execution": "next_close"},
        "exec_close": {"execution": "close"},
        "init_1N": {"_init_1N": True},
    }


def _env(cfg, market, featureset, rng_, reward_overrides, variant_env, scen):
    over = {**(variant_env or {}), **{k: v for k, v in scen.items() if not k.startswith("_")}}
    return make_eval_env(cfg, market, featureset, rng_, reward_overrides, over)


def _init(scen: Mapping[str, Any], n: int) -> np.ndarray | None:
    return np.full(n, 1.0 / n) if scen.get("_init_1N") else None


def eval_chain(cfg: Mapping[str, Any], variant, seed: int, root: str, scen_names: Sequence[str]) -> dict[str, Any]:
    """Evaluate one chain's fold models under every scenario; also replay on training windows."""
    market, featureset = prepare_data(cfg)
    ecfg = cfg["experiments"]
    folds = walk_forward_splits(
        market.n_periods, first_valid_index(featureset, cfg), train_window=int(ecfg["train_window"]),
        test_window=int(ecfg["test_window"]), step=int(ecfg["test_window"]),
        embargo=int(ecfg.get("embargo", 0)), expanding=False, n_folds=int(ecfg["n_folds"]),
        anchor=str(ecfg.get("fold_anchor", "end")))
    scen = scenarios(cfg)
    out: dict[str, list[pd.DataFrame]] = {s: [] for s in scen_names}
    clip_rows = []
    clip = float(cfg["reward"]["clip"])
    scale = float(cfg["reward"]["reward_scale"])
    for f in folds:
        path = Path(root) / slugify(variant.key) / f"seed_{seed}" / f"fold_{f.index:02d}" / "final_model.zip"
        er = f.evaluation_range(market.n_periods)
        policy = None
        for s in scen_names:
            env = _env(cfg, market, featureset, er, variant.reward_overrides, variant.env_overrides, scen[s])
            if policy is None:
                policy = load_policy(path, env, None, True)
            out[s].append(run_episode(env, action_fn=policy, initial_weights=_init(scen[s], market.n_assets))
                          .assign(fold=f.index, seed=seed))
        # Reward-clip replay on the training window, in consecutive training-length episodes.
        ep = int(cfg["environment"]["episode_length"])
        s0, s1 = f.train_range
        start = max(s0, first_valid_index(featureset, cfg))
        raw_vals = []
        for b in range(start, s1 - 2, ep):
            env = make_portfolio_env(
                featureset.values, market.close, market.volume, cfg, start_index=b, end_index=min(b + ep + 1, s1),
                overrides={**(variant.env_overrides or {}), "random_start": False, "episode_length": None,
                           "bankruptcy_threshold": 0.0},
                reward_overrides=variant.reward_overrides, asset_names=market.assets,
                dates=list(market.dates), open_prices=market.open)
            obs, _ = env.reset(options={"start_index": b})
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(policy(obs))
                raw_vals.append(scale * (info["reward_return_term"] - info["cvar_penalty"]
                                         - info["drawdown_penalty"] - info["turnover_penalty"]))
                done = term or trunc
        raw = np.asarray(raw_vals)
        clip_rows.append({"variant": variant.label, "seed": seed, "fold": f.index, "window": "train_replay",
                          "n": raw.size, "n_clipped": int((np.abs(raw) > clip).sum()),
                          "min_raw": float(raw.min()), "max_raw": float(raw.max())})
    return {"variant": variant.label, "seed": seed, "frames": out, "clip": clip_rows}


def series(by_seed: Mapping[int, pd.DataFrame], idx: pd.Index) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return ([d["return"].reindex(idx).to_numpy() for d in by_seed.values()],
            [d["turnover"].reindex(idx).to_numpy() for d in by_seed.values()])


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="outputs_real_nextopen")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="subset of seeds (smoke tests)")
    p.add_argument("--out-suffix", default="", help="suffix for output file names (smoke tests)")
    args = p.parse_args(argv)
    configure_logging()
    res = Path(args.results)
    cfg = load_config(res / "config_used.yaml")
    ecfg = cfg["experiments"]
    scen = scenarios(cfg)
    names = args.scenarios or list(scen)
    market, featureset = prepare_data(cfg)
    folds = walk_forward_splits(
        market.n_periods, first_valid_index(featureset, cfg), train_window=int(ecfg["train_window"]),
        test_window=int(ecfg["test_window"]), step=int(ecfg["test_window"]),
        embargo=int(ecfg.get("embargo", 0)), expanding=False, n_folds=int(ecfg["n_folds"]),
        anchor=str(ecfg.get("fold_anchor", "end")))
    t0 = time.perf_counter()

    # Baselines and the untrained control (main process).
    stitched: dict[str, dict[str, dict[int, pd.DataFrame]]] = {s: {} for s in names}
    baselines = build_baselines(ecfg.get("baselines", {}), float(cfg["environment"].get("max_gross_leverage", 1.0)))
    n = market.n_assets
    for s in names:
        for name, strat in baselines.items():
            frames = [run_episode(_env(cfg, market, featureset, f.evaluation_range(market.n_periods), None, None,
                                       scen[s]), weights_fn=strat, initial_weights=_init(scen[s], n))
                      .assign(fold=f.index) for f in folds]
            stitched[s][name] = {0: stitch(frames)}
        frames = [run_episode(_env(cfg, market, featureset, f.evaluation_range(market.n_periods), None, None,
                                   scen[s]), weights_fn=lambda e: np.full(n, 1.0 / (n + 1)),
                              initial_weights=_init(scen[s], n)).assign(fold=f.index) for f in folds]
        stitched[s][UNTRAINED] = {0: stitch(frames)}
    logger.info("Baselines done (%.1f min)", (time.perf_counter() - t0) / 60)

    variants = build_variants(ecfg, None)
    seeds = args.seeds or [int(x) for x in ecfg["seeds"]]
    jobs = [(v, s) for v in variants for s in seeds]
    clip_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(1,)) as pool:
        futs = [pool.submit(eval_chain, cfg, v, s, str(res / "models"), names) for v, s in jobs]
        for i, fut in enumerate(futs, 1):
            r = fut.result()
            for s in names:
                stitched[s].setdefault(r["variant"], {})[r["seed"]] = stitch(r["frames"][s])
            clip_rows += r["clip"]
            logger.info("[%d/%d] %s seed=%d (%.1f min)", i, len(jobs), r["variant"][:20], r["seed"],
                        (time.perf_counter() - t0) / 60)
    with (res / f"sensitivity_records{args.out_suffix}.pkl").open("wb") as fh:
        pickle.dump(stitched, fh)

    # Out-of-sample clip frequency (base scenario) for completeness.
    for label, by_seed in stitched["base"].items():
        if not label.startswith("PPO"):
            continue
        for sd, d in by_seed.items():
            clip_rows.append({"variant": label, "seed": sd, "fold": -1, "window": "oos", "n": len(d),
                              "n_clipped": int((d["reward"].abs() >= float(cfg["reward"]["clip"]) - 1e-12).sum()),
                              "min_raw": float(d["reward"].min()), "max_raw": float(d["reward"].max())})
    pd.DataFrame(clip_rows).to_csv(res / f"reward_clipping{args.out_suffix}.csv", index=False)

    # Level table: seed-mean metrics per strategy and scenario.
    rows = []
    for s in names:
        for label, by_seed in stitched[s].items():
            vals = {m: [] for m in METRICS}
            for d in by_seed.values():
                r, to = d["return"].to_numpy(), d["turnover"].to_numpy()
                for m, fn in METRICS.items():
                    vals[m].append(fn(r, to))
            cost = [100 * d["cost"].sum() for d in by_seed.values()]
            rows.append({"scenario": s, "strategy": label, "n_seeds": len(by_seed),
                         **{m: float(np.mean(v)) for m, v in vals.items()},
                         "sharpe_seed_sd": float(np.std(vals["sharpe"], ddof=1)) if len(by_seed) > 1 else np.nan,
                         "total_cost_pct_nav": float(np.mean(cost))})
    levels = pd.DataFrame(rows)
    levels.to_csv(res / f"sensitivity_levels{args.out_suffix}.csv", index=False)

    # Do the primary conclusions survive? Same comparisons as P1-P4 (net Sharpe), per scenario.
    labels = {v.key: v.label for v in variants}
    comps = [("P1", labels["E_full_calibrated"], labels["A_vanilla"]),
             ("P2", labels["S_smoothed"], labels["A_vanilla"]),
             ("P3", labels["SE_smoothed_calibrated"], EW),
             ("P4", labels["A_vanilla"], EW),
             ("S-1/N", labels["S_smoothed"], EW),
             ("SE-untrained", labels["SE_smoothed_calibrated"], UNTRAINED)]
    crow = []
    for s in names:
        st = stitched[s]
        idx = st[EW][0].index
        for cid, a, b in comps:
            ra, ta = series(st[a], idx)
            rb, tb = series(st[b], idx)
            c = two_level_bootstrap(ra, rb, "sharpe", ta, tb, n_boot=args.n_boot)
            crow.append({"scenario": s, "id": cid, "A": a, "B": b, "estimate": c.estimate, "ci_low": c.ci_low,
                         "ci_high": c.ci_high, "p_value": c.p_value, "mde80": minimum_detectable_effect(c.se)})
    comp = pd.DataFrame(crow)
    comp.to_csv(res / f"sensitivity_comparisons{args.out_suffix}.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 500):
        piv = levels.pivot_table(index="strategy", columns="scenario", values="sharpe", sort=False)
        print(piv[names].round(3).to_string())
        print(comp.assign(A=comp["A"].str[:14], B=comp["B"].str[:14]).round(3).to_string(index=False))
        cr = pd.DataFrame(clip_rows)
        print(cr.groupby(["variant", "window"])[["n", "n_clipped"]].sum().to_string())
    logger.info("Finished in %.1f min", (time.perf_counter() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
