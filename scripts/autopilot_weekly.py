"""Weekly autopilot: retrain calibration, evaluate NGR holdout, settle archive.

Runs as an unattended weekly job. Writes a timestamped evaluation report and
a one-line status summary to `logs/autopilot.log`. Safe to re-run; idempotent.

Usage (from repo root):
    python scripts/autopilot_weekly.py
    python scripts/autopilot_weekly.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_app_config, resolve_config_path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] autopilot: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
log = logging.getLogger("autopilot")


def _run(cmd: list[str], dry_run: bool) -> int:
    log.info("$ %s", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly autopilot retrain + evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Show commands, don't execute")
    parser.add_argument("--days", type=int, default=365, help="Training window")
    parser.add_argument("--holdout-days", type=int, default=30, help="Evaluation holdout")
    args = parser.parse_args()

    config = load_app_config(resolve_config_path())
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reports_dir = REPO_ROOT / "data" / "evaluation_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"ngr_autopilot_{stamp}.json"

    status: dict = {"stamp": stamp, "steps": {}}

    # 1) Settle opportunity archive + paper trades against latest actuals
    if config.get("opportunity_archive_enabled", True):
        rc = _run([PYTHON, "scripts/settle_opportunity_archive.py"], args.dry_run)
        status["steps"]["settle_archive"] = rc
    rc = _run([PYTHON, "main.py", "--settle-paper-trades"], args.dry_run)
    status["steps"]["settle_paper_trades"] = rc

    # 2) Retrain all calibration models (EMOS + isotonic + NGR)
    rc = _run([PYTHON, "train_calibration.py", "--days", str(args.days)], args.dry_run)
    status["steps"]["train_calibration"] = rc

    # 3) Evaluate NGR vs raw on chronological holdout
    rc = _run(
        [PYTHON, "scripts/evaluate_ngr.py",
         "--days", str(args.days + 30),
         "--holdout-days", str(args.holdout_days),
         "--output", str(report_path)],
        args.dry_run,
    )
    status["steps"]["evaluate_ngr"] = rc

    # 4) Read the report and decide whether NGR is still winning
    if not args.dry_run and report_path.exists():
        try:
            summary = json.loads(report_path.read_text())
            total = summary.get("pairs_evaluated", 0)
            better = summary.get("pairs_better_crps", 0)
            mean_ngr = summary.get("mean_crps_ngr") or 0.0
            mean_raw = summary.get("mean_crps_raw") or 0.0
            win_ratio = (better / total) if total else 0.0
            status["report"] = {
                "pairs_evaluated": total,
                "pairs_better_crps": better,
                "win_ratio": round(win_ratio, 3),
                "mean_crps_ngr": round(mean_ngr, 3),
                "mean_crps_raw": round(mean_raw, 3),
                "ngr_improvement_pct": round(
                    100.0 * (mean_raw - mean_ngr) / mean_raw, 1
                ) if mean_raw else None,
            }
            # Simple drift alarm: auto-disable NGR if win ratio drops below 0.5
            if total > 0 and win_ratio < 0.5:
                log.warning(
                    "NGR win ratio %.1f%% < 50%% — consider flipping use_ngr_calibration=false",
                    win_ratio * 100,
                )
                status["alert"] = "ngr_below_threshold"
            else:
                log.info(
                    "NGR holding: %d/%d pairs better CRPS (mean %.3f vs raw %.3f, %.1f%% improvement)",
                    better, total, mean_ngr, mean_raw,
                    status["report"].get("ngr_improvement_pct") or 0.0,
                )
        except Exception as exc:
            log.warning("Could not parse evaluation report: %s", exc)

    # 5) Write status log
    status_dir = REPO_ROOT / "logs"
    status_dir.mkdir(parents=True, exist_ok=True)
    with open(status_dir / "autopilot.log", "a") as f:
        f.write(json.dumps(status) + "\n")

    overall = max(status["steps"].values()) if status["steps"] else 0
    log.info("Autopilot weekly complete (exit=%d)", overall)
    return overall


if __name__ == "__main__":
    sys.exit(main())
