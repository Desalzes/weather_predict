# NGR Calibration Upgrade — Design

**Date:** 2026-04-16
**Status:** Design approved, pending spec review
**Scope:** Replace linear EMOS with Non-homogeneous Gaussian Regression (NGR), add opportunity-level logging, switch to EV-based sizing, loosen trading policy once the new calibration is validated.

---

## Motivation

Post-mortem of the three settled EMOS-route paper trades (Boston high 4/4, LA low 4/4, San Antonio high 4/4) shows a consistent pattern:

| # | Residual | σ used | Residual / σ |
|---|----------|--------|--------------|
| 1 | +3.86°F | 1.85 | 2.1 |
| 2 | −3.76°F | **1.00 (floor)** | 3.8 |
| 3 | −6.40°F | 1.91 | 3.4 |

The mean corrections from linear EMOS were small and directionally fine (+0.3, 0.0, −1.6°F). **The losses came from σ being too tight, not μ being wrong.** The current pipeline passes `ensemble_std_f` through a hard `max(σ, 1.0)` floor; that 1.0°F floor was the direct cause of LA's 70× calibration lift on a near-impossible event that then happened.

Isotonic calibration masked the problem by pulling small raw probabilities up (0.0008 → 0.055 on trade #2; 0.087 → 0.287 on trade #3). With better σ, isotonic has less work to do and we expect fewer tail blowups.

Separately, the project generates almost no ground-truth signal per day: v3 policy produced 0 paper trades on the first VPS run, and even at its peak yields 0–3 trades/scan. We cannot evaluate calibration improvements on that volume.

---

## Goals

1. **Predictive accuracy** — replace linear-mean EMOS with joint (μ, σ) NGR using season, lead time, and ensemble spread as features.
2. **Data volume** — log every scored opportunity (not just traded ones) so calibration can be evaluated on hundreds of examples per day, not 0–3.
3. **Profitability** — once NGR is validated, loosen the policy modestly and introduce quarter-Kelly sizing so conviction translates into position size.

All three must ship to move the needle. Shipping only #1 without #2 means we can't validate #1; shipping only #2 and #3 without #1 means we're sizing up on mis-calibrated probabilities.

---

## Non-Goals

- No auto-retraining pipeline. Manual `train_calibration.py` runs remain the default.
- No per-(city, market) edge thresholds. Global threshold + sizing formula handle this.
- No new forecast sources (NDFD, ECMWF raw, GFS direct). Open-Meteo + HRRR only.
- No GOES cloud adjustment (separate work item).
- No dashboard integration.
- No live-trading changes. Paper trading only, as today.

---

## Model Specification

### Functional form (per city × market_type)

```
μ  = α₀ + α₁·f + α₂·s + α₃·sin(2π·d/365) + α₄·cos(2π·d/365) + α₅·ℓ
σ² = max( σ²_floor,  β₀ + β₁·s² + β₂·ℓ )
```

Where:

| Symbol | Meaning |
|--------|---------|
| `f` | Raw forecast value (°F) — daily high or low from Open-Meteo best-match |
| `s` | Ensemble spread (°F) — `ensemble_{high,low}_std_f` from archive |
| `d` | Day-of-year of the **target date** (integer 1–366) |
| `ℓ` | Lead in hours, `(target_date_close_time − as_of_utc) / 3600` |
| `σ²_floor` | Empirical 10th-percentile of historical squared residuals for this (city, market_type) |

Six μ parameters, three σ parameters, one data-driven floor. Nine total free parameters per pair. With ~90 days × 1–3 lead-time archive rows per day ≈ 100–300 rows per (city, market), this is appropriately constrained.

### Training objective

Minimize **mean CRPS** for a Gaussian predictive distribution (closed form, Gneiting et al. 2005):

```
CRPS(N(μ,σ²), y) = σ · [ z·(2Φ(z) − 1) + 2·φ(z) − 1/√π ]
where z = (y − μ) / σ
```

CRPS directly rewards both sharpness and calibration. Mean-squared-error (what the current linear EMOS minimizes) is blind to σ. Switching the objective is itself part of the upgrade.

Implementation: `scipy.optimize.minimize(method="L-BFGS-B")`. Analytic gradients available; we'll start with numerical and add analytic only if fit time is a problem (expected <1s per pair).

### Fallbacks

- If training rows < 20 for a pair → skip NGR, fall back to current linear EMOS.
- If NGR fit fails (optimizer non-convergence, NaN in data) → skip NGR, fall back to linear EMOS, log a warning.
- If no EMOS either → raw forecast passthrough, as today.
- Selective-raw-fallback list (5 pairs) kept but revalidated under NGR on holdout.

### Persistence

New artifact: `data/calibration_models/{city}_{market_type}_ngr.pkl` containing a `NGRCalibrator` instance. Existing `_emos.pkl` and `_isotonic.pkl` files remain in place and continue to load for the fallback path. No breaking change to on-disk model registry.

---

## Architecture Changes

### New: `NGRCalibrator` class in `src/calibration.py`

```python
@dataclass
class NGRCalibrator:
    city: str
    market_type: str
    alpha: np.ndarray  # (6,) mean coefficients
    beta: np.ndarray   # (3,) variance coefficients
    sigma2_floor: float
    training_rows: int
    training_crps: float
    is_fitted: bool

    def fit(self, df: pd.DataFrame) -> "NGRCalibrator": ...
    def predict(self, forecast_f, spread_f, lead_h, doy) -> tuple[float, float]:
        """Returns (mu, sigma)."""
```

### Changed: `CalibrationManager`

New primary method:

```python
def predict_distribution(
    self,
    city: str,
    market_type: str,
    forecast_f: float,
    spread_f: float,
    lead_h: float,
    doy: int,
) -> tuple[float, float, str]:
    """Returns (mu, sigma, source) where source ∈ {'ngr', 'emos', 'raw', 'raw_selective_fallback'}."""
```

Existing `correct_forecast()` is retained but reimplemented as a thin wrapper that calls `predict_distribution` and discards σ. This preserves the old test surface during migration.

### Changed: `src/matcher.py`

Replace the current pattern:

```python
sigma_f, sigma_source = _resolve_temperature_uncertainty(...)  # ensemble → clamp
forecast_value, calibration_source = _apply_calibration(...)   # μ only
```

With:

```python
ensemble_sigma_f = _ensemble_sigma_for_date(...)  # unclamped raw input
lead_h = (close_dt_utc - now_utc).total_seconds() / 3600
doy = datetime.fromisoformat(market_date).timetuple().tm_yday
mu, sigma_f, calibration_source = calibration_manager.predict_distribution(
    city, mtype, raw_forecast_value, ensemble_sigma_f, lead_h, doy
)
```

The ensemble σ becomes a **feature**, not the final σ. The `configure_sigma_bounds()` hooks remain for EMOS fallback path and for boundary safety (hard `σ ≥ 0.25°F, σ ≤ 12°F` sanity clamps post-NGR), but the active-path floor is now per-pair and data-driven.

### New: `src/opportunity_log.py`

```python
def log_opportunities(
    scan_id: str,
    scored_opportunities: list[dict],
    archive_dir: Path,
) -> Path:
    """Append all scored opportunities to data/opportunity_archive/YYYY-MM-DD.csv."""

def settle_opportunity_archive(
    archive_dir: Path,
    station_actuals: dict,
) -> int:
    """Join archived opps with station truth; write yes_outcome column in place."""
```

Schema per row: `scan_id, recorded_at_utc, source, ticker, city, market_type, market_date, outcome, our_probability, raw_probability, market_price, edge, abs_edge, mu_ngr, sigma_ngr, mu_raw, spread_raw, lead_h, doy, forecast_blend_source, forecast_calibration_source, probability_calibration_source, volume24hr, yes_outcome, actual_value_f, settled_at_utc`.

**Important:** This is write-only. It does not feed back into trading decisions. It exists to enable honest calibration evaluation on real market pricing over hundreds of labeled events per day.

Settlement of the archive runs as part of `main.py --settle-paper-trades` (same pass that handles paper trades).

### Changed: `train_calibration.py`

Adds NGR fit alongside existing EMOS + isotonic pipeline. Output summary includes per-pair training CRPS and a comparison against the old EMOS pipeline's CRPS on the same data.

### Changed: `evaluate_calibration.py`

Adds NGR as a third policy alongside `raw` and `selective_fallback`. Holdout comparison tracks CRPS (new), Brier, MAE, and per-pair regressions. The existing `--compare` flag works unchanged.

### New config keys (`config.example.json`)

| Key | Default | Purpose |
|-----|---------|---------|
| `use_ngr_calibration` | `false` (Phase 1–2) → `true` (Phase 3) | Master flag for NGR path in matcher |
| `ngr_min_training_rows` | `20` | Minimum rows to attempt NGR fit |
| `ngr_sigma_floor_quantile` | `0.10` | Quantile used to derive σ²_floor |
| `opportunity_archive_enabled` | `true` | Enable the opportunity log |
| `opportunity_archive_dir` | `data/opportunity_archive` | Archive location |
| `kelly_fraction` | `0.25` | Fraction of full-Kelly to size at |
| `bankroll_dollars` | `100.0` | Notional bankroll for sizing (paper) |
| `max_contracts_hard_cap` | `20` | Absolute ceiling regardless of Kelly output |

### Changed: `strategy/strategy_policy.json` (v4, gated on Phase 4)

```json
{
  "policy_version": 4,
  "selection": {
    "sources": ["kalshi"],
    "min_abs_edge": 0.08,
    "min_volume24hr": 2000,
    "max_candidates_per_scan": 8,
    "max_hours_to_settlement": 48,
    "allowed_market_types": ["high", "low"],
    "allowed_settlement_rules": ["lte", "gt"],
    "allowed_cities": [],
    "blocked_cities": ["Washington DC"]
  },
  "execution": {
    "sizing": "quarter_kelly",
    "max_contracts_per_trade": 20,
    "max_new_orders_per_day": 10,
    "max_order_cost_dollars": 10.0,
    "time_in_force": "fill_or_kill"
  }
}
```

v3 is retained as `strategy/strategy_policy_v3.json` for rollback.

### Changed: `src/paper_trading.py`

New helper:

```python
def compute_position_size(
    edge: float,
    price: float,
    kelly_fraction: float,
    bankroll_dollars: float,
    max_order_cost_dollars: float,
    hard_cap_contracts: int,
) -> int:
    """Quarter-Kelly contract sizing for binary-payout markets."""
```

Formula (exact Kelly for a binary YES/NO contract with payoff 1):

```
# BUY side (paying `price` for YES, our_prob = p)
kelly_full = (p − price) / (1 − price)

# SELL side (collecting `price` on NO, our_prob = p means NO_prob = 1 − p)
kelly_full = ((1 − p) − (1 − price)) / price      # == (price − p) / price

stake_$    = kelly_fraction · bankroll · max(0, kelly_full)
contracts  = min(
    floor(stake_$ / effective_price),              # effective_price = price on BUY, (1−price) on SELL
    floor(max_order_cost_dollars / effective_price),
    hard_cap_contracts,
)
```

`edge / (price · (1 − price))` is a log-return approximation; not used here.

Sizing is also logged per trade so we can analyze it in post-hoc.

---

## Data Flow

```
forecast_archive + station_actuals
        │
        ▼
   training frame  (forecast_f, spread_f, lead_h, doy, actual_f)
        │
        ▼
   fit NGR  ──►  {city}_{type}_ngr.pkl
        │
        ▼
   evaluate_calibration.py (CRPS / Brier on holdout vs EMOS)
        │
        ▼
   flip use_ngr_calibration = true

---- live path ----

   Open-Meteo + HRRR forecast
        │
        ▼
   matcher:
      ensemble_σ  ──► NGRCalibrator.predict_distribution(μ, σ, ℓ, d)
                           │
                           ▼
                      P(outcome) via normal CDF
        │
        ▼
   opportunity_log.log_opportunities(all_scored_opps)  ←─ every scan
        │
        ▼
   strategy_policy v4 filter
        │
        ▼
   paper_trading.compute_position_size(edge, price)  ←─ quarter-Kelly
        │
        ▼
   ledger (with contracts, mu_ngr, sigma_ngr columns)
```

---

## Rollout Plan

Each phase gated by a config flag; each reversible.

### Phase 1 — Foundation (no behavior change)
- Ship `NGRCalibrator`, `opportunity_log`, and training updates.
- `use_ngr_calibration=false` in prod config.
- Live trades still use legacy EMOS + isotonic.
- **Exit criterion:** training runs cleanly on existing archive, all existing tests pass, new unit tests green.

### Phase 2 — Shadow mode (read-only NGR in live path)
- Every scan computes both legacy and NGR probabilities; writes both to opportunity archive.
- Decisions still use legacy.
- Run for ≥3 days (or ≥500 archived opportunities per city, whichever first).
- **Exit criterion:** NGR shows no pathological outputs (no σ < 0.25°F or > 12°F; no μ drift > 10°F from raw), and holdout CRPS ≤ legacy CRPS on ≥70% of pairs.

### Phase 3 — NGR as live primary
- Flip `use_ngr_calibration=true`.
- Policy v3 still in place.
- **Exit criterion:** ≥3 days AND either (a) ≥10 settled NGR-route trades with ROI ≥ legacy-route ROI, or (b) <10 settled trades but holdout CRPS for NGR remains ≥ legacy on ≥70% of pairs.

### Phase 4 — Policy v4 + Kelly sizing
- Swap to `strategy_policy.json` v4.
- Turn on `kelly_fraction=0.25` in execution.
- **Exit criterion:** None — this is the state we want to settle into.

Rollback at any phase: flip config flag, redeploy. No data migration.

---

## Testing Strategy

### Unit tests
- **NGR fitting** on synthetic data with known (α, β) — verify recovery within tolerance.
- **NGR prediction** — σ responds to spread, lead time, and floor correctly.
- **CRPS closed form** — numerical check against Monte Carlo CRPS.
- **Kelly sizing** — corner cases: edge=0 → 0 contracts; edge>0 at price=0.01 → capped by max_order_cost; edge>0 at price=0.99 → hard_cap applies.
- **Opportunity log** — round-trip write/read/settle.

### Integration tests
- Full matcher flow with NGR enabled vs disabled, same inputs.
- Backwards compat: old `.pkl` files still load and work.
- `train_calibration.py` on a slice of real archive produces NGR artifacts.
- `evaluate_calibration.py` reports NGR alongside raw and selective_fallback.

### Evaluation tests (non-unit, gates Phase 2→3 and Phase 3→4)
- Holdout CRPS delta per pair.
- Brier delta per pair.
- Reliability diagram on archived opportunities after settlement pass.

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| NGR overfits with only ~100 rows per pair | Medium | Fallback to linear EMOS at <20 rows; log training CRPS alongside OOS CRPS; freeze selective fallback list as safety net |
| Optimizer fails to converge on some pairs | Medium | Try multiple initializations; fall back to linear EMOS on failure; log + alert |
| Kelly sizes up too aggressively on a mispriced estimate | Medium-low | Quarter-Kelly, hard cap of 20 contracts, `max_order_cost_dollars=10` still applies |
| Opportunity archive grows unbounded | Low | Daily files; estimated ~500 rows/day × ~200 bytes = 100 KB/day; 40 MB/year; fine |
| Policy v4 floods paper trade book before validation | Low | Phase 4 is gated on Phase 3 success; kelly_fraction starts at 0.25; rollback is config flag |
| NGR σ collapses to floor on some pair | Medium | σ²_floor derived from data not magic number; unit test asserts σ ≥ floor; monitoring log on any scan where σ = floor for >20% of opps |

---

## Files Touched (summary)

**New:**
- `src/opportunity_log.py`
- `tests/test_ngr_calibration.py`
- `tests/test_opportunity_log.py`
- `tests/test_kelly_sizing.py`
- `data/opportunity_archive/` (runtime, gitignored initially)
- `strategy/strategy_policy_v3.json` (backup of current)

**Modified:**
- `src/calibration.py` — add `NGRCalibrator`, extend `CalibrationManager`
- `src/matcher.py` — switch to `predict_distribution`, pass lead_h and doy
- `src/paper_trading.py` — add `compute_position_size`, wire into trade recording
- `train_calibration.py` — fit NGR, report CRPS comparison
- `evaluate_calibration.py` — add NGR policy in holdout comparison
- `main.py` — wire opportunity logging, wire settlement pass
- `config.example.json` — new config keys
- `strategy/strategy_policy.json` — v4 payload (deferred activation)
- Existing `test_calibration.py`, `test_matcher.py`, `test_strategy_policy.py` — update for new call signatures

---

## Open Questions

None at design time. All parameters (Kelly fraction, bankroll, edge thresholds, σ floor quantile) are config-driven and can be tuned after first weeks of paper data.
