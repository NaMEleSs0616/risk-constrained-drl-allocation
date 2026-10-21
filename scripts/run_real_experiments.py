#!/usr/bin/env python
"""Walk-forward study on market data with warm-started fine-tuning across folds.

Design (see config/real_etf.yaml):

* Many rolling folds (default 14 x 252-day test windows). Each (variant, seed) is a
  *chain*: fold 0 is trained from scratch, every later fold fine-tunes the previous
  fold's model on its own rolling training window. Only data before each test window
  is ever used for training.
* Every fold is evaluated with the model as it stands at the end of that fold's
  training - no validation-based checkpoint selection (which interacted with the
  penalties in the synthetic study).
* Variants can carry fixed penalties, a turnover penalty, RCPO adaptive multipliers,
  and a post-hoc volatility-targeting overlay (evaluated on the same models).
* Classical baselines and the untrained-policy control run through the same simulator.

Usage:
    python scripts/run_real_experiments.py --config config/real_etf.yaml
    python scripts/run_real_experiments.py --config config/real_etf.yaml --resume
    python scripts/run_real_experiments.py --config config/real_etf.yaml --smoke
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.run_experiments import (  # noqa: E402
    Variant,
    _init_worker,
    aggregate,
    build_variants,
    slugify,
    write_artifacts,
)
from src.backtest import Fold, load_policy, make_eval_env, run_episode, walk_forward_splits  # noqa: E402
from src.baselines import VolTargetOverlay, build_baselines  # noqa: E402
from src.train import first_valid_index, prepare_data, train_agent  # noqa: E402
from src.utils import (  # noqa: E402
    configure_logging,
    deep_update,
    get_logger,
    load_config,
    parse_overrides,
    resolve_path,
    save_config,
)

logger = get_logger("run_real_experiments")

SMOKE: dict[str, Any] = {
    "ppo": {"n_envs": 2, "n_steps": 128, "batch_size": 64, "n_epochs": 2},
    "experiments": {"n_folds": 3, "initial_timesteps": 512, "finetune_timesteps": 256, "seeds": [0],
                    "n_workers": 2},
}


@dataclass(frozen=True)
class ChainJob:
    """One (variant, seed) chain over all folds."""

    variant: Variant
    seed: int
    folds: tuple[Fold, ...]
    root: str
    initial_timesteps: int
    finetune_timesteps: int


def ew_constraint_targets(market, train_range: tuple[int, int], scale: float = 1.0,
                          window: int = 63, alpha: float = 0.95, episode: int = 252) -> tuple[float, float]:
    """RCPO targets matched to the 1/N portfolio's risk inside a training window (no look-ahead).

    Returns:
        ``(cvar_target, drawdown_target)``: the 1/N portfolio's average rolling daily CVaR and
        its average squared drawdown over consecutive ``episode``-day blocks (each block starts
        from a fresh peak, as training episodes do), both multiplied by ``scale``.
    """
    from src.reward import drawdown_series, historical_cvar

    s, e = train_range
    px = market.close[s:e]
    r = (px[1:] / px[:-1] - 1.0).mean(axis=1)
    cvars = [historical_cvar(r[i - window:i], alpha) for i in range(window, len(r) + 1)]
    dd2 = []
    for b in range(0, len(r), episode):
        nav = np.concatenate([[1.0], np.cumprod(1.0 + r[b:b + episode])])
        dd2.append(np.mean(drawdown_series(nav) ** 2))
    return scale * float(np.mean(cvars)), scale * max(float(np.mean(dd2)), 1e-5)


def overlay_label(v: Variant) -> str:
    return f"{v.label} + vol target {v.vol_target_overlay:.0%}"


def run_chain(cfg: Mapping[str, Any], job: ChainJob, deterministic: bool, resume: bool) -> dict[str, Any]:
    """Train a chain fold by fold and evaluate each fold's final model out of sample.

    Runs in a worker process and never raises.

    Returns:
        ``{"job", "records": {label: [fold frames]}, "seconds", "error"}``.
    """
    t0 = time.perf_counter()
    records: dict[str, list[pd.DataFrame]] = {job.variant.label: []}
    if job.variant.vol_target_overlay:
        records[overlay_label(job.variant)] = []
    try:
        market, featureset = prepare_data(cfg)
        P = int(cfg["environment"].get("periods_per_year", 252))
        prev: Path | None = None
        for fold in job.folds:
            run_dir = Path(job.root) / slugify(job.variant.key) / f"seed_{job.seed}" / f"fold_{fold.index:02d}"
            final = run_dir / "final_model.zip"
            lagr = job.variant.lagrangian
            if lagr and "ew_target_scale" in lagr:
                c_t, d_t = ew_constraint_targets(market, fold.train_range, lagr["ew_target_scale"])
                lagr = {**lagr, "cvar_target": c_t, "drawdown_target": d_t}
                (run_dir).mkdir(parents=True, exist_ok=True)
                (run_dir / "rcpo_targets.json").write_text(
                    __import__("json").dumps({"cvar_target": c_t, "drawdown_target": d_t}))
            if not (resume and final.exists()):
                result = train_agent(
                    cfg, market, featureset, fold.train_range, run_dir,
                    val_range=None,
                    total_timesteps=job.initial_timesteps if prev is None else job.finetune_timesteps,
                    reward_overrides=job.variant.reward_overrides,
                    init_model_path=prev,
                    run_name=f"{job.variant.key}_s{job.seed}_f{fold.index}",
                    seed=1000 * job.seed + fold.index,
                    log_stdout=False,
                    lagrangian=lagr,
                    save_checkpoints=False,
                    env_overrides=job.variant.env_overrides,
                )
                final = result.model_path
            prev = final
            eval_range = fold.evaluation_range(market.n_periods)
            env = make_eval_env(cfg, market, featureset, eval_range, job.variant.reward_overrides,
                                job.variant.env_overrides)
            policy = load_policy(final, env, None, deterministic)
            records[job.variant.label].append(
                run_episode(env, action_fn=policy).assign(fold=fold.index, seed=job.seed))
            if job.variant.vol_target_overlay:
                env = make_eval_env(cfg, market, featureset, eval_range, job.variant.reward_overrides,
                                    job.variant.env_overrides)
                overlay = VolTargetOverlay(policy, job.variant.vol_target_overlay, periods_per_year=P)
                records[overlay_label(job.variant)].append(
                    run_episode(env, weights_fn=overlay).assign(fold=fold.index, seed=job.seed))
        return {"job": job, "records": records, "seconds": time.perf_counter() - t0, "error": None}
    except Exception:  # noqa: BLE001
        return {"job": job, "records": None, "seconds": time.perf_counter() - t0, "error": traceback.format_exc()}


def run(cfg: Mapping[str, Any], resume: bool = False, variant_keys: Sequence[str] | None = None) -> None:
    """Execute the market-data walk-forward study and write all artifacts."""
    ecfg = cfg["experiments"]
    out = resolve_path(ecfg.get("output_dir", "outputs_real"))
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config_used.yaml")
    P = int(cfg["environment"].get("periods_per_year", 252))

    market, featureset = prepare_data(cfg)
    test_window = int(ecfg["test_window"])
    folds = walk_forward_splits(
        market.n_periods, first_valid_index(featureset, cfg),
        train_window=int(ecfg["train_window"]), test_window=test_window, step=test_window,
        embargo=int(ecfg.get("embargo", 0)), expanding=False, n_folds=int(ecfg["n_folds"]),
    )
    logger.info("%d folds: first test %s, last test end %s", len(folds),
                market.dates[folds[0].test_start].date(), market.dates[folds[-1].test_end - 1].date())

    variants = build_variants(ecfg, variant_keys)
    seeds = [int(s) for s in ecfg.get("seeds", [0])]
    jobs = [ChainJob(v, s, tuple(folds), str(out / "models"), int(ecfg["initial_timesteps"]),
                     int(ecfg["finetune_timesteps"])) for v in variants for s in seeds]
    steps = len(jobs) * (int(ecfg["initial_timesteps"]) + (len(folds) - 1) * int(ecfg["finetune_timesteps"]))
    logger.info("%d chains (%d variants x %d seeds), %s PPO steps in total", len(jobs), len(variants),
                len(seeds), f"{steps:,}")

    # Baselines and the untrained-policy control (deterministic, main process).
    baseline_frames: dict[str, list[pd.DataFrame]] = {}
    baselines = build_baselines(ecfg.get("baselines", {}), float(cfg["environment"].get("max_gross_leverage", 1.0)))
    n = market.n_assets
    for f in folds:
        er = f.evaluation_range(market.n_periods)
        for name, strat in baselines.items():
            baseline_frames.setdefault(name, []).append(
                run_episode(make_eval_env(cfg, market, featureset, er), weights_fn=strat).assign(fold=f.index))
        baseline_frames.setdefault("Untrained policy (1/(N+1) incl. cash)", []).append(
            run_episode(make_eval_env(cfg, market, featureset, er),
                        weights_fn=lambda e: np.full(n, 1.0 / (n + 1))).assign(fold=f.index))
    logger.info("Evaluated %d baselines + untrained control on %d folds", len(baselines), len(folds))

    t0 = time.perf_counter()
    progress = out / "progress.log"
    results: list[dict[str, Any]] = []
    n_workers = min(int(ecfg.get("n_workers", 1)), len(jobs))
    threads = int(ecfg.get("torch_threads", 2))
    det = bool(ecfg.get("deterministic", True))

    def report(res: dict[str, Any]) -> None:
        results.append(res)
        j: ChainJob = res["job"]
        status = "ok" if res["error"] is None else "FAILED"
        line = (f"[{len(results)}/{len(jobs)}] {status} {j.variant.key} seed={j.seed} "
                f"({res['seconds'] / 60:.1f} min) | elapsed {(time.perf_counter() - t0) / 60:.1f} min")
        logger.info(line)
        with progress.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if res["error"]:
            logger.error("Chain failed:\n%s", res["error"])

    if n_workers <= 1:
        _init_worker(threads)
        configure_logging(logging.INFO)
        for j in jobs:
            report(run_chain(cfg, j, det, resume))
    else:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker, initargs=(threads,)) as pool:
            for fut in as_completed([pool.submit(run_chain, cfg, j, det, resume) for j in jobs]):
                report(fut.result())

    rl_frames: dict[str, dict[int, list[pd.DataFrame]]] = {}
    for v in variants:
        rl_frames[v.label] = {}
        if v.vol_target_overlay:
            rl_frames[overlay_label(v)] = {}
    failures = []
    for r in results:
        j = r["job"]
        if r["records"] is None:
            failures.append(f"{j.variant.key} seed={j.seed}")
            continue
        for label, frames in r["records"].items():
            if len(frames) == len(folds):
                rl_frames[label][j.seed] = frames
    regimes = pd.Series(market.regimes, index=market.dates) if market.regimes is not None else None
    reference = variants[0].label if variants and variants[0].key == "A_vanilla" else None
    res = aggregate(rl_frames, baseline_frames, folds, regimes, P, float(ecfg.get("risk_free_rate", 0.0)), reference)
    res.failures = failures
    print("\n" + write_artifacts(res, out, market.dates, regimes))
    logger.info("Finished in %.1f min (%d failed chains); artifacts in %s",
                (time.perf_counter() - t0) / 60, len(failures), out)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/real_etf.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args(argv)
    configure_logging()
    over = deep_update(SMOKE if args.smoke else {}, parse_overrides(args.set))
    if args.output_dir:
        over = deep_update(over, {"experiments": {"output_dir": args.output_dir}})
    run(load_config(resolve_path(args.config), over), args.resume, args.variants)
    return 0


if __name__ == "__main__":
    sys.exit(main())
