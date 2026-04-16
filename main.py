"""
Weather Signals — Market Opportunity Scanner

Fetches Open-Meteo forecasts for 20 US cities, then compares against live
Kalshi and Polymarket weather markets to find mispriced temperature bets.

Usage:
    python main.py --once          # single scan
    python main.py                 # continuous loop
    python main.py --kalshi-only   # Kalshi only
    python main.py --poly-only     # Polymarket only
"""

import argparse
import logging
import math
import os
import time
from datetime import datetime, timezone

from src.config import load_app_config, resolve_config_path
from src.logging_setup import configure_logging, set_log_level

_CONFIG_PATH = str(resolve_config_path())
log = configure_logging("weather", config_path=_CONFIG_PATH, log_filename="weather.log", level=logging.INFO)


def load_config(config_path: str = _CONFIG_PATH) -> dict:
    return load_app_config(config_path)


def _print_banner(config: dict) -> None:
    locations = config.get("locations", [])
    interval = config.get("scan_interval_minutes", 30)
    min_edge = config.get("min_edge_threshold", 0.05)
    kalshi = config.get("enable_kalshi", True)
    poly = config.get("enable_polymarket", True)
    sources = []
    if kalshi: sources.append("Kalshi")
    if poly: sources.append("Polymarket")
    log.info("=" * 60)
    log.info("  Weather Signals - Market Opportunity Scanner")
    log.info("  Locations: %d cities", len(locations))
    log.info("  Scan interval: %d min", interval)
    log.info("  Min edge: %.0f%%", min_edge * 100)
    log.info("  Sources: %s", " + ".join(sources) or "none")
    log.info("=" * 60)


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _select_hrrr_requirements(
    locations: list,
    markets: list[dict],
    *,
    time_key: str,
    horizon_hours: float,
    now_utc: datetime,
) -> tuple[list[dict], int]:
    if horizon_hours <= 0:
        return [], 0

    by_name = {loc.get("name"): loc for loc in locations if loc.get("name")}
    selected = {}
    max_hours_to_settlement = 0.0

    for market in markets:
        city = market.get("city")
        if city not in by_name:
            continue

        close_time = _parse_iso_datetime(str(market.get(time_key, "")))
        if close_time is None:
            continue

        hours_to_settlement = (close_time.astimezone(timezone.utc) - now_utc).total_seconds() / 3600.0
        if 0 <= hours_to_settlement <= horizon_hours:
            selected[city] = by_name[city]
            max_hours_to_settlement = max(max_hours_to_settlement, hours_to_settlement)

    if not selected:
        return [], 0

    return list(selected.values()), max(1, int(math.ceil(max_hours_to_settlement)))


def _merge_hrrr_locations(*location_groups: list[dict]) -> list[dict]:
    merged = {}
    for locations in location_groups:
        for loc in locations:
            name = loc.get("name")
            if name:
                merged[name] = loc
    return list(merged.values())


def run_scan(
    config: dict = None,
    skip_kalshi: bool = False,
    skip_polymarket: bool = False,
) -> dict:
    """Execute a single scan pass."""
    if config is None:
        config = load_config()

    timestamp = datetime.now(timezone.utc).isoformat()
    locations = config.get("locations", [])
    forecast_hours = config.get("forecast_hours", 72)
    min_edge = config.get("min_edge_threshold", 0.05)
    uncertainty = config.get("uncertainty_std_f", 2.0)
    enable_ensemble = config.get("enable_ensemble", False)
    enable_calibration = config.get("enable_calibration", False)
    enable_hrrr = config.get("enable_hrrr", False)
    enable_paper_trading = config.get("enable_paper_trading", False)
    paper_trade_auto_settle = config.get("paper_trade_auto_settle", True)
    paper_trade_contracts = float(config.get("paper_trade_contracts", 1.0))
    paper_trade_ledger_path = config.get("paper_trade_ledger_path")
    paper_trade_summary_path = config.get("paper_trade_summary_path")
    calibration_model_dir = config.get("calibration_model_dir")
    hrrr_blend_horizon_hours = float(config.get("hrrr_blend_horizon_hours", 18))
    now_utc = datetime.now(timezone.utc)

    # Apply configurable constants to matcher and calibration modules
    try:
        from src.matcher import configure_sigma_bounds
        configure_sigma_bounds(
            floor_f=config.get("ensemble_sigma_floor_f"),
            cap_f=config.get("ensemble_sigma_cap_f"),
        )
    except ImportError:
        pass
    try:
        from src.calibration import configure_calibration_constants
        configure_calibration_constants(
            min_spread_f=config.get("emos_min_spread_f"),
            selective_raw_fallback_targets=config.get("selective_raw_fallback_targets"),
        )
    except ImportError:
        pass

    log.info("Fetching forecasts for %d locations (%dh horizon)...", len(locations), forecast_hours)
    from src.fetch_forecasts import fetch_ensemble_multi_location, fetch_multi_location
    forecasts = fetch_multi_location(locations, forecast_hours)
    valid = {k: v for k, v in forecasts.items() if v is not None}
    log.info("Got forecasts for %d/%d locations.", len(valid), len(locations))
    ensemble_data = {}

    if enable_ensemble:
        try:
            log.info(
                "Fetching ensemble forecasts for %d locations (%dh horizon)...",
                len(locations),
                forecast_hours,
            )
            ensemble_data = fetch_ensemble_multi_location(locations, forecast_hours)
            ensemble_valid = {k: v for k, v in ensemble_data.items() if v is not None}
            log.info("Got ensemble forecasts for %d/%d locations.", len(ensemble_valid), len(locations))
        except Exception as exc:
            log.warning("Ensemble fetch failed; using fallback uncertainty %.1fF: %s", uncertainty, exc)
            ensemble_data = {}

    try:
        from src.station_truth import archive_forecast_snapshot

        archived = 0
        for city, forecast in valid.items():
            archive_forecast_snapshot(city, forecast, ensemble_data.get(city))
            archived += 1
        log.info("Archived forecast snapshots for %d/%d locations.", archived, len(valid))
    except Exception as exc:
        log.warning("Forecast archive step failed: %s", exc)

    calibration_manager = None
    if enable_calibration:
        try:
            from src.calibration import CalibrationManager

            calibration_manager = CalibrationManager(model_dir=calibration_model_dir)
            log.info(
                "Calibration enabled with %d model file(s).",
                calibration_manager.available_model_count(),
            )
        except Exception as exc:
            log.warning("Calibration setup failed; using raw forecasts/probabilities: %s", exc)
            calibration_manager = None

    all_opportunities = []
    kalshi_markets = []
    poly_markets = []
    kalshi_hrrr_data = {}
    poly_hrrr_data = {}

    if not skip_kalshi and config.get("enable_kalshi", True):
        try:
            from src.fetch_kalshi import fetch_weather_markets as fetch_kalshi

            log.info("Scanning Kalshi weather markets...")
            kalshi_markets = fetch_kalshi(pages=3)
            log.info("Found %d Kalshi weather markets.", len(kalshi_markets))
        except Exception as exc:
            log.warning("Kalshi scan failed: %s", exc)
            kalshi_markets = []

    if not skip_polymarket and config.get("enable_polymarket", True):
        try:
            from src.fetch_polymarket import fetch_weather_markets as fetch_poly

            log.info("Scanning Polymarket weather markets...")
            poly_markets = fetch_poly()
            log.info("Found %d Polymarket weather markets.", len(poly_markets))
        except Exception as exc:
            log.warning("Polymarket scan failed: %s", exc)
            poly_markets = []

    if enable_hrrr:
        try:
            from src.fetch_hrrr import fetch_hrrr_multi

            kalshi_hrrr_locations, kalshi_hrrr_fxx = _select_hrrr_requirements(
                locations,
                kalshi_markets,
                time_key="close_time",
                horizon_hours=hrrr_blend_horizon_hours,
                now_utc=now_utc,
            )
            poly_hrrr_locations, poly_hrrr_fxx = _select_hrrr_requirements(
                locations,
                poly_markets,
                time_key="endDate",
                horizon_hours=hrrr_blend_horizon_hours,
                now_utc=now_utc,
            )

            if kalshi_hrrr_locations:
                log.info(
                    "Kalshi HRRR targets: %d location(s) through f%02d.",
                    len(kalshi_hrrr_locations),
                    kalshi_hrrr_fxx,
                )
            if poly_hrrr_locations:
                log.info(
                    "Polymarket HRRR targets: %d location(s) through f%02d.",
                    len(poly_hrrr_locations),
                    poly_hrrr_fxx,
                )

            shared_hrrr_locations = _merge_hrrr_locations(kalshi_hrrr_locations, poly_hrrr_locations)
            shared_hrrr_fxx = max(kalshi_hrrr_fxx, poly_hrrr_fxx)
            if shared_hrrr_locations and shared_hrrr_fxx > 0:
                log.info(
                    "Fetching shared HRRR point forecasts for %d unique location(s) through f%02d...",
                    len(shared_hrrr_locations),
                    shared_hrrr_fxx,
                )
                shared_hrrr_data = fetch_hrrr_multi(
                    shared_hrrr_locations,
                    fxx=shared_hrrr_fxx,
                    now=now_utc,
                    timeout_seconds=config.get("hrrr_timeout_seconds"),
                    max_retries_per_hour=config.get("hrrr_max_retries_per_hour"),
                    retry_delay_seconds=config.get("hrrr_retry_delay_seconds"),
                )
                if kalshi_hrrr_locations:
                    kalshi_hrrr_data = shared_hrrr_data
                if poly_hrrr_locations:
                    poly_hrrr_data = shared_hrrr_data
            else:
                log.info("No markets fall within the HRRR blend horizon.")
        except RuntimeError as exc:
            log.warning("HRRR unavailable; install Herbie stack to enable Phase 4: %s", exc)
        except Exception as exc:
            log.warning("Shared HRRR fetch failed: %s", exc)

    if kalshi_markets:
        try:
            from src.matcher import match_kalshi_markets

            kalshi_opps = match_kalshi_markets(
                valid,
                kalshi_markets,
                min_edge=min_edge,
                uncertainty_std_f=uncertainty,
                ensemble_data=ensemble_data,
                calibration_manager=calibration_manager,
                hrrr_data=kalshi_hrrr_data,
                hrrr_blend_horizon_hours=hrrr_blend_horizon_hours,
                now_utc=now_utc,
                use_ngr_calibration=bool(config.get("use_ngr_calibration", False)),
            )
            all_opportunities.extend(kalshi_opps)
            log.info("Kalshi: %d opportunities with edge >= %.0f%%", len(kalshi_opps), min_edge * 100)
        except Exception as exc:
            log.warning("Kalshi match failed: %s", exc)

    if poly_markets:
        try:
            from src.matcher import match_polymarket_markets

            poly_opps = match_polymarket_markets(
                valid,
                poly_markets,
                min_edge=min_edge,
                uncertainty_std_f=uncertainty,
                ensemble_data=ensemble_data,
                calibration_manager=calibration_manager,
                hrrr_data=poly_hrrr_data,
                hrrr_blend_horizon_hours=hrrr_blend_horizon_hours,
                now_utc=now_utc,
                use_ngr_calibration=bool(config.get("use_ngr_calibration", False)),
            )
            all_opportunities.extend(poly_opps)
            log.info("Polymarket: %d opportunities with edge >= %.0f%%", len(poly_opps), min_edge * 100)
        except Exception as exc:
            log.warning("Polymarket match failed: %s", exc)

    all_opportunities.sort(key=lambda x: x.get("abs_edge", 0), reverse=True)

    if config.get("opportunity_archive_enabled", True):
        try:
            from src.opportunity_log import log_opportunities
            log_opportunities(
                scan_id=timestamp,
                opportunities=all_opportunities,
                archive_dir=config.get("opportunity_archive_dir", "data/opportunity_archive"),
            )
        except Exception as exc:
            log.warning("Opportunity archive write failed: %s", exc)

    from src.matcher import format_report
    report = format_report(all_opportunities)
    print("\n" + report)

    # Always apply strategy policy filter to narrow trade candidates
    from src.strategy_policy import filter_opportunities_for_policy, load_strategy_policy

    strategy_policy, strategy_policy_path = load_strategy_policy(config.get("strategy_policy_path"))
    trade_opportunities = filter_opportunities_for_policy(all_opportunities, strategy_policy)
    log.info(
        "Strategy policy candidates: %d/%d from %s.",
        len(trade_opportunities),
        len(all_opportunities),
        strategy_policy_path,
    )

    deepseek_review = None
    if config.get("enable_deepseek_worker", False):
        try:
            from src.deepseek_worker import (
                approved_opportunities_from_review,
                deepseek_trade_mode,
                format_review_summary,
                maybe_run_deepseek_review,
            )

            deepseek_review = maybe_run_deepseek_review(
                trade_opportunities,
                config=config,
                scan_timestamp=timestamp,
                strategy_policy=strategy_policy,
            )
            log.info("%s", format_review_summary(deepseek_review))

            if deepseek_trade_mode(config) == "paper_gate":
                review_status = str(deepseek_review.get("status", "")).strip().lower()
                review_reason = str(deepseek_review.get("reason", "")).strip().lower()
                if review_status == "completed":
                    trade_opportunities = approved_opportunities_from_review(trade_opportunities, deepseek_review)
                    log.info(
                        "DeepSeek paper gate approved %d/%d opportunities for paper execution.",
                        len(trade_opportunities),
                        len(filter_opportunities_for_policy(all_opportunities, strategy_policy)),
                    )
                elif review_status == "skipped" and review_reason == "interval_not_reached":
                    log.info(
                        "DeepSeek review was skipped by interval gate; using %d policy-filtered opportunities for paper execution.",
                        len(trade_opportunities),
                    )
                else:
                    trade_opportunities = []
                    log.info(
                        "DeepSeek paper gate withheld paper trades for this scan because review status was %s.",
                        deepseek_review.get("status"),
                    )
        except Exception as exc:
            log.warning("DeepSeek worker failed: %s", exc)
            deepseek_review = {
                "status": "failed",
                "reason": "integration_error",
                "error": str(exc),
                "scan_timestamp": timestamp,
            }
            if str(config.get("deepseek_trade_mode", "")).strip().lower() == "paper_gate":
                trade_opportunities = []

    paper_trade_log = None
    paper_trade_summary = None
    if enable_paper_trading:
        try:
            from src.paper_trading import (
                format_paper_trade_summary,
                log_paper_trades,
                refresh_station_actuals_for_open_trades,
                settle_paper_trades,
            )

            paper_trade_log = log_paper_trades(
                trade_opportunities,
                scan_timestamp=timestamp,
                ledger_path=paper_trade_ledger_path,
                contracts=paper_trade_contracts,
            )
            log.info(
                "Paper trade ledger: %d new trade(s), %d duplicate open position(s) skipped.",
                paper_trade_log["new_trades"],
                paper_trade_log["skipped_existing_open_positions"],
            )
            if paper_trade_auto_settle:
                refresh_result = refresh_station_actuals_for_open_trades(
                    config=config,
                    ledger_path=paper_trade_ledger_path,
                )
                if refresh_result.get("attempted"):
                    log.info(
                        "Paper trade station-actual refresh: %d city window(s) via %s through %s.",
                        len(refresh_result.get("city_windows", [])),
                        refresh_result.get("refresh_source"),
                        refresh_result.get("cutoff_date"),
                    )
                paper_trade_summary = settle_paper_trades(
                    ledger_path=paper_trade_ledger_path,
                    summary_path=paper_trade_summary_path,
                )
                log.info("%s", format_paper_trade_summary(paper_trade_summary))
        except Exception as exc:
            log.warning("Paper trade logging/settlement failed: %s", exc)

    log.info("Scan complete: %d total opportunities.", len(all_opportunities))

    return {
        "timestamp": timestamp,
        "locations_analyzed": len(valid),
        "opportunities": all_opportunities,
        "trade_opportunities": trade_opportunities,
        "deepseek_review": deepseek_review,
        "paper_trade_log": paper_trade_log,
        "paper_trade_summary": paper_trade_summary,
    }


def run_loop(config: dict = None, **kwargs) -> None:
    if config is None:
        config = load_config()

    _print_banner(config)
    interval_seconds = config.get("scan_interval_minutes", 30) * 60

    log.info("Starting continuous loop (interval=%ds). Ctrl+C to stop.", interval_seconds)

    while True:
        try:
            run_scan(config=config, **kwargs)
        except Exception as exc:
            log.error("Scan iteration failed: %s", exc, exc_info=True)

        log.info("Next scan in %d seconds.", interval_seconds)
        try:
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            log.info("Interrupted. Exiting.")
            break


def run_paper_trade_settlement(
    config: dict = None,
    *,
    as_of_date: str | None = None,
) -> dict:
    if config is None:
        config = load_config()

    from src.paper_trading import (
        format_paper_trade_summary,
        refresh_station_actuals_for_open_trades,
        settle_paper_trades,
    )

    refresh_station_actuals_for_open_trades(
        config=config,
        ledger_path=config.get("paper_trade_ledger_path"),
        as_of_date=as_of_date,
    )

    summary = settle_paper_trades(
        ledger_path=config.get("paper_trade_ledger_path"),
        summary_path=config.get("paper_trade_summary_path"),
        as_of_date=as_of_date,
    )

    if config.get("opportunity_archive_enabled", True):
        try:
            from src.opportunity_log import settle_opportunity_archive
            from src.station_truth import STATION_ACTUALS_DIR, load_station_map
            import pandas as pd
            from pathlib import Path

            station_actuals_map = {}
            station_map = load_station_map()
            actuals_dir = Path(STATION_ACTUALS_DIR)
            for city in station_map:
                slug = city.lower().replace(" ", "_").replace(".", "").replace(",", "")
                p = actuals_dir / f"{slug}.csv"
                if p.exists():
                    df = pd.read_csv(p)
                    if {"date", "tmax_f", "tmin_f"}.issubset(df.columns):
                        station_actuals_map[city] = df
            archive_dir = config.get("opportunity_archive_dir", "data/opportunity_archive")
            count = settle_opportunity_archive(archive_dir=archive_dir, station_actuals=station_actuals_map)
            log.info("Opportunity archive: settled %d rows", count)
        except Exception as exc:
            log.warning("Opportunity archive settlement failed: %s", exc)

    print(format_paper_trade_summary(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather Signals - Market Opportunity Scanner")
    parser.add_argument("--once", action="store_true", help="Single scan and exit.")
    parser.add_argument("--config", default=_CONFIG_PATH, help="Config path.")
    parser.add_argument("--kalshi-only", action="store_true", help="Kalshi only, skip Polymarket.")
    parser.add_argument("--poly-only", action="store_true", help="Polymarket only, skip Kalshi.")
    parser.add_argument(
        "--settle-paper-trades",
        action="store_true",
        help="Settle logged paper trades from saved station actuals and exit.",
    )
    parser.add_argument(
        "--as-of-date",
        help="Optional YYYY-MM-DD settlement cutoff for --settle-paper-trades.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args()

    if args.verbose:
        set_log_level(logging.DEBUG)

    config = load_config(args.config)

    skip_kalshi = args.poly_only
    skip_polymarket = args.kalshi_only

    if args.settle_paper_trades:
        run_paper_trade_settlement(config=config, as_of_date=args.as_of_date)
        return

    if args.once:
        _print_banner(config)
        run_scan(config=config, skip_kalshi=skip_kalshi, skip_polymarket=skip_polymarket)
    else:
        run_loop(config=config, skip_kalshi=skip_kalshi, skip_polymarket=skip_polymarket)


if __name__ == "__main__":
    main()
