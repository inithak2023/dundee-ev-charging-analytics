"""
EV CHARGING DEMAND, QUEUE PREDICTION & RECOMMENDATION
Revised end-to-end research pipeline implementing the supervisor's corrections.

INPUT
  One Dundee City Council charging-session CSV (one row per charging session).

OUTPUT
  results/ : CSV tables for revised IEEE Tables II, III, VI and supporting results
  figs/    : plots

IMPORTANT
  1. Download the CSV resource from the Dundee dataset portal and set CSV_PATH below.
  2. Check the detected columns printed at startup. If any are wrong, fill COLUMN_MAP.
  3. Distances in the recommendation experiment are RANDOM (0.5-10 km), not postcode-derived.
     State this explicitly in the abstract/methodology unless you add real geocoded distances.
  4. The Dundee catalogue currently lists no licence. Obtain confirmation from Dundee City
     Council before claiming OGL or another licence in the paper.
"""

from __future__ import annotations
import os, re, heapq, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

SEED = 42
np.random.seed(SEED)

# =============================================================================
# 0. CONFIGURATION -- EDIT THIS SECTION
# =============================================================================
CSV_PATH = os.environ.get("EV_CSV_PATH", "sessions.csv")  # private local CSV
OUTPUT_DIR = Path("ev_revised_outputs")
RESULTS_DIR = OUTPUT_DIR / "results"
FIGS_DIR = OUTPUT_DIR / "figs"

# Fill these only if automatic column detection chooses incorrectly.
# Use exact column names from your CSV. Leave None for auto-detection.
COLUMN_MAP = {
    "station": None,     # e.g. "CP ID" or "Charge Point"
    "start": None,       # e.g. "Start Date/Time"
    "end": None,         # e.g. "End Date/Time"
    "energy": None,      # e.g. "Energy (kWh)"
    "postcode": None     # optional; not used unless you implement geocoding
}

TRAIN_FRACTION = 0.80
CAPACITY_FACTORS = [0.4, 0.6, 0.8, 1.0, 1.2]
PRIMARY_CAPACITY_FACTOR = 0.6
DURATION_CAP_HOURS = 24.0
N_RECOMMENDATION_TRIALS = 2000
BOOTSTRAP_REPS = 5000
BLOCK_LENGTH_HOURS = 168
DISTANCE_MIN_KM = 0.5
DISTANCE_MAX_KM = 10.0
DISTANCE_WEIGHTS = [1.0, 2.0, 4.0]   # minutes per km
RANDOM_DISTANCES = True              # explicitly disclosed in outputs
MAX_CLUSTER_SAMPLE = 20000
EXPORT_SESSION_LEVEL_CLEANED = False  # keep session-level data local/private

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)
# =============================================================================
# 1. GENERAL HELPERS
# =============================================================================
def safe_r2(y, pred):
    return float(r2_score(y, pred)) if len(np.unique(np.asarray(y))) > 1 else np.nan

def regression_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    nonzero = y != 0
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAPE_percent": (
            np.mean(np.abs(y[nonzero] - pred[nonzero]) / np.abs(y[nonzero])) * 100
            if nonzero.any() else np.nan
        ),
        "R2": safe_r2(y, pred)
    }

def normalise_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def choose_column(columns, candidates, exclude=()):
    norm = {c: normalise_name(c) for c in columns}
    for key in candidates:
        matches = [c for c in columns if key in norm[c]
                   and not any(x in norm[c] for x in exclude)]
        if matches:
            return matches[0]
    return None

def parse_datetime(series):
    """Parse UK-style date/time robustly; choose parse with fewer missing values."""
    a = pd.to_datetime(series, errors="coerce", dayfirst=True)
    b = pd.to_datetime(series, errors="coerce", dayfirst=False)
    out = a if a.isna().sum() <= b.isna().sum() else b
    # Standardise timezone-aware values to naive local wall time, if present.
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out

def bootstrap_mean_ci(values, reps=BOOTSTRAP_REPS, seed=SEED):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    r = np.random.default_rng(seed)
    means = np.empty(reps)
    for b in range(reps):
        means[b] = r.choice(x, size=len(x), replace=True).mean()
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))

def block_bootstrap_mean_ci(values, block_len=BLOCK_LENGTH_HOURS,
                            reps=BOOTSTRAP_REPS, seed=SEED):
    """Circular moving-block bootstrap CI for a mean time-series statistic."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    L = max(1, min(int(block_len), n))
    r = np.random.default_rng(seed)
    out = np.empty(reps)
    n_blocks = int(np.ceil(n / L))
    starts_max = max(1, n)
    for b in range(reps):
        starts = r.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([x[(s + np.arange(L)) % n] for s in starts])[:n]
        out[b] = sample.mean()
    return float(x.mean()), float(np.quantile(out, .025)), float(np.quantile(out, .975))


# =============================================================================
# 2. LOAD AND STANDARDISE RAW SESSION DATA
# =============================================================================
def detect_columns(raw):
    cols = list(raw.columns)
    station = COLUMN_MAP["station"] or choose_column(
        cols, ["chargepointid", "chargepoint", "stationid", "stationname",
               "evseid", "chargerid", "siteid", "station", "site", "location"]
    )
    energy = COLUMN_MAP["energy"] or choose_column(
        cols, ["energykwh", "kwh", "energy"]
    )
    def datetime_spec(side):
        # Prefer a single combined timestamp column; otherwise combine date + time.
        if COLUMN_MAP[side] is not None:
            return COLUMN_MAP[side]
        ncols = {c: normalise_name(c) for c in cols}
        combined_keys = (["startdatetime", "connectionstartdatetime", "plugindatetime"]
                         if side == "start" else
                         ["enddatetime", "connectionenddatetime", "disconnectdatetime"])
        for key in combined_keys:
            hit = next((c for c in cols if key in ncols[c]), None)
            if hit:
                return hit
        date_keys = (["startdate", "connectiondate", "plugindate"] if side == "start"
                     else ["enddate", "disconnectdate", "endconnectiondate"])
        time_keys = (["starttime", "connectionstart", "plugintime"] if side == "start"
                     else ["endtime", "connectionend", "disconnecttime"])
        date_col = next((c for c in cols if any(k in ncols[c] for k in date_keys)), None)
        time_col = next((c for c in cols if any(k in ncols[c] for k in time_keys)
                         and c != date_col), None)
        if date_col and time_col:
            return [date_col, time_col]
        # Fall back to any plausible single start/end field; user can override.
        fallback_keys = date_keys + time_keys + (["start"] if side == "start" else ["end", "stop"])
        return next((c for c in cols if any(k in ncols[c] for k in fallback_keys)), None)

    start = datetime_spec("start")
    end = datetime_spec("end")
    postcode = COLUMN_MAP["postcode"] or choose_column(
        cols, ["postcode", "postalcode", "zipcode"]
    )
    return station, start, end, energy, postcode

def load_sessions(csv_path):
    raw = pd.read_csv(csv_path, low_memory=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    print("\nCSV columns:\n", list(raw.columns))
    station_col, start_col, end_col, energy_col, postcode_col = detect_columns(raw)
    detected = {
        "station": station_col, "start": start_col, "end": end_col,
        "energy": energy_col, "postcode": postcode_col
    }
    print("\nDetected columns:", detected)
    required = [station_col, start_col, end_col, energy_col]
    if any(x is None for x in required):
        raise ValueError(
            "Could not detect all required columns. Edit COLUMN_MAP at the top "
            "using the exact names printed above."
        )
    def combine_datetime(spec):
        if isinstance(spec, (list, tuple)):
            values = raw[spec[0]].astype(str).str.strip() + " " + raw[spec[1]].astype(str).str.strip()
            return parse_datetime(values)
        return parse_datetime(raw[spec])

    d = pd.DataFrame({
        "station": raw[station_col].astype(str).str.strip(),
        "start": combine_datetime(start_col),
        "end": combine_datetime(end_col),
        "energy_kwh": pd.to_numeric(raw[energy_col], errors="coerce")
    })
    if postcode_col:
        d["postcode"] = raw[postcode_col].astype(str).str.strip()
    else:
        d["postcode"] = np.nan

    d = d.dropna(subset=["station", "start", "end", "energy_kwh"])
    d = d.drop_duplicates().copy()
    d["duration_h_original"] = (d["end"] - d["start"]).dt.total_seconds() / 3600
    d = d[(d["duration_h_original"] > 0) & (d["energy_kwh"] >= 0)].copy()
    d = d.sort_values(["station", "start", "end"]).reset_index(drop=True)
    d["session_id"] = np.arange(len(d))
    d["hour_ts"] = d["start"].dt.floor("h")
    d["power_kw"] = d["energy_kwh"] / d["duration_h_original"].replace(0, np.nan)
    d["power_kw"] = d["power_kw"].replace([np.inf, -np.inf], np.nan)
    return d, detected


# =============================================================================
# 3. DATASET INSPECTION AND 24-HOUR CLEANING
# =============================================================================
def make_original_and_cleaned(data):
    d0 = data.copy()
    d0["duration_h"] = d0["duration_h_original"]
    d1 = data.copy()
    d1["duration_h"] = d1["duration_h_original"].clip(upper=DURATION_CAP_HOURS)
    d1["end_effective"] = d1["start"] + pd.to_timedelta(d1["duration_h"], unit="h")
    d0["end_effective"] = d0["start"] + pd.to_timedelta(d0["duration_h"], unit="h")

    over = data["duration_h_original"] > DURATION_CAP_HOURS
    total_hours = data["duration_h_original"].sum()
    over_hours = data.loc[over, "duration_h_original"].sum()
    stats = pd.DataFrame([{
        "total_sessions": len(data),
        "sessions_over_24h": int(over.sum()),
        "sessions_over_24h_percent": over.mean() * 100 if len(data) else np.nan,
        "charger_hours_all_sessions_original": total_hours,
        "charger_hours_from_over_24h_sessions": over_hours,
        "share_charger_hours_from_over_24h_percent":
            over_hours / total_hours * 100 if total_hours else np.nan,
        "mean_duration_original_h": data["duration_h_original"].mean(),
        "mean_duration_after_cap_h": d1["duration_h"].mean(),
        "total_charger_hours_after_cap": d1["duration_h"].sum()
    }])
    stats.to_csv(RESULTS_DIR / "table_dataset_cleaning_24h.csv", index=False)
    if EXPORT_SESSION_LEVEL_CLEANED:
        d1.to_csv(RESULTS_DIR / "sessions_cleaned_cap24h.csv", index=False)
    return d0, d1, stats


# =============================================================================
# 4. DEMAND FORECASTING: FIVE EXPANDING-WINDOW PERIODS
# =============================================================================
def build_hourly_demand(data):
    x = data.set_index("start").sort_index()
    hourly = x.resample("h").size().rename("sessions")
    full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_idx, fill_value=0).astype(float)
    hourly.index.name = "hour_ts"
    return hourly.to_frame()

def make_demand_features(hourly_df):
    f = hourly_df.copy()
    f["hour"] = f.index.hour
    f["dow"] = f.index.dayofweek
    f["weekend"] = (f["dow"] >= 5).astype(int)
    for lag in [1, 24, 168]:
        f[f"lag{lag}"] = f["sessions"].shift(lag)
    f["roll24"] = f["sessions"].shift(1).rolling(24).mean()
    return f.dropna()

def historical_mean_predict(train, test):
    temp = train.copy()
    temp["dow"] = temp.index.dayofweek
    temp["hour"] = temp.index.hour
    means = temp.groupby(["dow", "hour"])["sessions"].mean()
    return np.array([
        means.get((idx.dayofweek, idx.hour), train["sessions"].mean())
        for idx in test.index
    ], dtype=float)

def fit_ets_weekly(train_series, horizon):
    """Fit additive weekly ETS; reject fits that report convergence warnings."""
    import warnings
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    model = ExponentialSmoothing(
        train_series.astype(float),
        trend=None,
        seasonal="add",
        seasonal_periods=168,
        initialization_method="heuristic"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = model.fit(optimized=True, use_brute=True)

    convergence_warnings = [
        w for w in caught
        if issubclass(w.category, ConvergenceWarning)
    ]

    if convergence_warnings:
        raise RuntimeError(
            "ETS optimisation did not converge: "
            + "; ".join(str(w.message) for w in convergence_warnings)
        )

    # Some statsmodels versions expose the optimiser status.
    mle_retvals = getattr(fitted, "mle_retvals", None)
    if isinstance(mle_retvals, dict) and mle_retvals.get("success") is False:
        raise RuntimeError("ETS optimiser returned success=False.")

    forecast = np.asarray(fitted.forecast(horizon), dtype=float)
    if not np.isfinite(forecast).all():
        raise RuntimeError("ETS produced non-finite forecast values.")

    return np.maximum(0, forecast)

def expanding_window_forecasting(hourly):
    f = make_demand_features(hourly)
    features = ["hour", "dow", "weekend", "lag1", "lag24", "lag168", "roll24"]
    splitter = TimeSeriesSplit(n_splits=5)
    all_records = []
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(splitter.split(f), start=1):
        tr, te = f.iloc[tr_idx].copy(), f.iloc[te_idx].copy()
        ytr, yte = tr["sessions"].values, te["sessions"].values
        pred = {}

        pred["Historical Mean"] = historical_mean_predict(tr, te)
        # Same hour one week earlier: lag 168.
        pred["Seasonal Naive (168h)"] = te["lag168"].values.astype(float)

        rf = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1,
            random_state=SEED
        ).fit(tr[features], ytr)
        pred["Random Forest"] = np.maximum(0, rf.predict(te[features]))

        xgb = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", n_jobs=-1,
            random_state=SEED
        ).fit(tr[features], ytr)
        pred["XGBoost"] = np.maximum(0, xgb.predict(te[features]))

        try:
            pred["ETS weekly (168h)"] = fit_ets_weekly(
                hourly.loc[:tr.index[-1], "sessions"], len(te)
            )
        except Exception as e:
            print(f"ETS failed on fold {fold}: {e}")
            pred["ETS weekly (168h)"] = np.full(len(te), np.nan)

        for model_name, values in pred.items():
            for timestamp, actual, estimate in zip(te.index, yte, values):
                all_records.append({
                    "fold": fold, "timestamp": timestamp,
                    "model": model_name, "actual": actual, "prediction": estimate
                })
            m = regression_metrics(yte, values) if np.isfinite(values).all() else {}
            fold_metrics.append({"fold": fold, "model": model_name, **m})

    pred_df = pd.DataFrame(all_records)
    pred_df.to_csv(RESULTS_DIR / "forecast_expanding_window_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(
        RESULTS_DIR / "forecast_metrics_by_fold.csv", index=False
    )

    overall = []
    for name, g in pred_df.groupby("model"):
        g = g.dropna(subset=["prediction"])
        overall.append({"model": name, **regression_metrics(g.actual, g.prediction)})
    table = pd.DataFrame(overall).set_index("model").sort_values("MAE")
    table.to_csv(RESULTS_DIR / "table_II_demand_forecasting.csv")

    # Block-bootstrap paired difference: XGBoost MAE minus Historical Mean MAE.
    wide = pred_df.pivot_table(
        index=["fold", "timestamp"], columns="model", values=["actual", "prediction"]
    )
    actual = wide[("actual", "XGBoost")].values
    xgb_pred = wide[("prediction", "XGBoost")].values
    hm_pred = wide[("prediction", "Historical Mean")].values
    valid = np.isfinite(actual) & np.isfinite(xgb_pred) & np.isfinite(hm_pred)
    delta_abs = np.abs(actual[valid] - xgb_pred[valid]) - np.abs(actual[valid] - hm_pred[valid])
    point, lo, hi = block_bootstrap_mean_ci(delta_abs, BLOCK_LENGTH_HOURS)
    pd.DataFrame([{
        "difference_definition": "MAE_XGBoost - MAE_Historical_Mean",
        "mean_MAE_difference": point,
        "block_bootstrap_95CI_lower": lo,
        "block_bootstrap_95CI_upper": hi,
        "block_length_hours": BLOCK_LENGTH_HOURS,
        "bootstrap_repetitions": BOOTSTRAP_REPS
    }]).to_csv(RESULTS_DIR / "forecast_xgb_vs_historical_block_bootstrap.csv", index=False)

    # Lag correlations for the corrected IV-A claim.
    lag_rows = []
    for lag in [1, 24, 168]:
        lag_rows.append({
            "lag_hours": lag,
            "correlation_with_current_demand": hourly["sessions"].corr(
                hourly["sessions"].shift(lag)
            )
        })
    pd.DataFrame(lag_rows).to_csv(
        RESULTS_DIR / "table_lag_correlations.csv", index=False
    )

    # Plot actual vs XGB across out-of-fold test periods.
    xg = pred_df[pred_df.model == "XGBoost"].dropna().sort_values("timestamp")
    plt.figure(figsize=(11, 4))
    plt.plot(xg.timestamp, xg.actual, label="Actual", linewidth=1)
    plt.plot(xg.timestamp, xg.prediction, label="XGBoost", linewidth=1)
    plt.legend()
    plt.title("Expanding-window demand forecasting: XGBoost")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "forecast_expanding_window_xgb.png", dpi=200)
    plt.close()
    table["MAE"].sort_values().plot.bar(figsize=(7, 4), title="Demand forecasting MAE")
    plt.ylabel("MAE (sessions/hour)")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "forecast_model_mae.png", dpi=200)
    plt.close()
    print("\nTABLE II - EXPANDING WINDOW FORECASTING\n", table.round(4))
    print("\nXGBoost vs Historical Mean block-bootstrap MAE difference:",
          point, "95% CI", (lo, hi))
    return table, pred_df


# =============================================================================
# 5. FCFS QUEUE REPLAY AND WAITING-TIME METRICS
# =============================================================================
def max_concurrency(group):
    events = []
    for row in group.itertuples():
        s = pd.Timestamp(row.start)
        e = s + pd.Timedelta(hours=float(row.duration_h))
        # -1 departures sort before +1 arrivals at identical timestamps.
        events.append((e.value, -1))
        events.append((s.value, +1))
    events.sort(key=lambda z: (z[0], z[1]))
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return max(1, int(peak))

def replay_fcfs(data, capacity_factor):
    d = data.sort_values(["station", "start", "session_id"]).copy()
    cap_obs = d.groupby("station", sort=False).apply(max_concurrency)
    cap_sim = (cap_obs * capacity_factor).round().clip(lower=1).astype(int)
    waits = np.zeros(len(d), dtype=float)
    row_pos = {idx: pos for pos, idx in enumerate(d.index)}
    for station, g in d.groupby("station", sort=False):
        c = int(cap_sim.loc[station])
        first = pd.Timestamp(g["start"].min())
        free = [first] * c
        heapq.heapify(free)
        for idx, row in g.iterrows():
            arrival = pd.Timestamp(row["start"])
            earliest_free = heapq.heappop(free)
            service_start = max(arrival, earliest_free)
            waits[row_pos[idx]] = max(
                0.0, (service_start - arrival).total_seconds() / 60.0
            )
            finish = service_start + pd.Timedelta(hours=float(row["duration_h"]))
            heapq.heappush(free, finish)
    d["wait_min"] = waits
    d["capacity_observed_max"] = d["station"].map(cap_obs).astype(int)
    d["capacity_simulated"] = d["station"].map(cap_sim).astype(int)
    d["hour_ts"] = d["start"].dt.floor("h")
    return d, cap_obs, cap_sim

def waiting_summary(d):
    waited = d.loc[d.wait_min > 0, "wait_min"]
    return {
        "sessions": len(d),
        "sessions_waiting": int((d.wait_min > 0).sum()),
        "sessions_waiting_percent": (d.wait_min > 0).mean() * 100,
        "mean_wait_all_sessions_min": d.wait_min.mean(),
        "mean_wait_among_waiting_min": waited.mean() if len(waited) else 0.0,
        "median_wait_among_waiting_min": waited.median() if len(waited) else 0.0,
        "max_wait_min": d.wait_min.max()
    }


# =============================================================================
# 6. QUEUE-PREDICTION DATA, VALIDATION-BASED MODEL SELECTION
# =============================================================================
QUEUE_FEATURES = [
    "arrivals_lag1", "wait_lag1",
    "arrivals_lag24", "wait_lag24", "capacity",
    "hour", "day_of_week", "station_code"
]

def make_station_hour_table(queued, cap_sim):
    """Build a complete hourly station panel before calculating past-only lags.

    Empty station-hours receive zero arrivals and zero mean wait. Rows with no
    current arrivals are removed only after lag features have been calculated.
    """
    if queued.empty:
        raise ValueError("Queue replay returned no sessions; cannot build station-hour data.")

    start_hour = pd.Timestamp(queued["hour_ts"].min()).floor("h")
    end_hour = pd.Timestamp(queued["hour_ts"].max()).floor("h")
    hours = pd.date_range(start_hour, end_hour, freq="h")
    stations = sorted(queued["station"].astype(str).unique())
    full_index = pd.MultiIndex.from_product(
        [stations, hours], names=["station", "hour_ts"]
    )

    observed = queued.groupby(["station", "hour_ts"]).agg(
        arrivals=("wait_min", "size"),
        target_wait=("wait_min", "mean")
    )
    sh = observed.reindex(full_index).reset_index()
    sh["arrivals"] = sh["arrivals"].fillna(0).astype(int)
    sh["target_wait"] = sh["target_wait"].fillna(0.0)
    sh["capacity"] = sh["station"].map(cap_sim).astype(float)
    sh["hour"] = sh["hour_ts"].dt.hour
    sh["day_of_week"] = sh["hour_ts"].dt.dayofweek
    station_codes = {name: i for i, name in enumerate(stations)}
    sh["station_code"] = sh["station"].map(station_codes).astype(int)

    sh = sh.sort_values(["station", "hour_ts"]).reset_index(drop=True)
    grouped = sh.groupby("station", sort=False)
    sh["arrivals_lag1"] = grouped["arrivals"].shift(1)
    sh["wait_lag1"] = grouped["target_wait"].shift(1)
    sh["arrivals_lag24"] = grouped["arrivals"].shift(24)
    sh["wait_lag24"] = grouped["target_wait"].shift(24)
    sh[QUEUE_FEATURES] = (
        sh[QUEUE_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # The target is defined only for station-hours with at least one arrival.
    sh = sh.loc[sh["arrivals"] > 0].copy()
    return sh.sort_values(["hour_ts", "station"]).reset_index(drop=True)

def make_queue_model(name):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", n_jobs=-1, random_state=SEED
        )
    raise ValueError(name)

def sakasegawa_prediction(frame, mean_service_hours, max_train_wait):
    c = np.maximum(1, frame["capacity"].values.astype(float))
    lam = frame["arrivals"].values.astype(float)
    rho = np.clip(lam * mean_service_hours / c, 0, 0.99)
    pred = (
        rho ** (np.sqrt(2 * (c + 1)) - 1)
        / np.maximum(c * (1 - rho), 1e-9)
        * mean_service_hours * 60
    )
    return np.minimum(pred, max_train_wait)

def train_validate_test_queue(sh, train_fraction=TRAIN_FRACTION):
    """Chronological train/validation/test split by unique hourly timestamps."""
    if sh.empty:
        raise ValueError("Station-hour training table is empty.")

    unique_times = np.array(sorted(pd.to_datetime(sh["hour_ts"].unique())))
    if len(unique_times) < 3:
        raise ValueError(
            f"Need at least 3 unique timestamps for chronological queue evaluation; got {len(unique_times)}."
        )

    cut_pos = int(len(unique_times) * train_fraction)
    cut_pos = min(max(cut_pos, 2), len(unique_times) - 1)
    test_start = unique_times[cut_pos]
    train_all = sh.loc[sh["hour_ts"] < test_start].copy()
    test = sh.loc[sh["hour_ts"] >= test_start].copy()

    train_times = np.array(sorted(pd.to_datetime(train_all["hour_ts"].unique())))
    if len(train_times) < 2 or train_all.empty or test.empty:
        raise ValueError("Insufficient chronological data for queue train/validation/test split.")

    # Reserve the final 10% of the pre-test training period for model selection.
    val_pos = int(np.floor(len(train_times) * 0.90))
    val_pos = min(max(val_pos, 1), len(train_times) - 1)
    val_start = train_times[val_pos]
    train = train_all.loc[train_all["hour_ts"] < val_start].copy()
    val = train_all.loc[train_all["hour_ts"] >= val_start].copy()
    if train.empty or val.empty:
        raise ValueError("Chronological queue validation split produced an empty partition.")

    candidates = ["Linear Regression", "Random Forest", "XGBoost"]
    validation_rows = []
    for name in candidates:
        model = make_queue_model(name)
        model.fit(train[QUEUE_FEATURES], train["target_wait"])
        pred = np.maximum(0, model.predict(val[QUEUE_FEATURES]))
        validation_rows.append({
            "model": name,
            **regression_metrics(val["target_wait"], pred)
        })

    validation_table = pd.DataFrame(validation_rows).sort_values("RMSE").reset_index(drop=True)
    selected = str(validation_table.iloc[0]["model"])
    selected_val_rmse = float(
        validation_table.loc[validation_table["model"] == selected, "RMSE"].iloc[0]
    )

    # Refit the validation-selected model on all pre-test observations.
    final_model = make_queue_model(selected)
    final_model.fit(train_all[QUEUE_FEATURES], train_all["target_wait"])
    test = test.copy()
    test["pred_wait"] = np.maximum(0, final_model.predict(test[QUEUE_FEATURES]))

    return {
        "selected_model": selected,
        "selected_model_validation_RMSE": selected_val_rmse,
        "validation_table": validation_table,
        "train_all": train_all,
        "test": test,
        "model": final_model
    }


# =============================================================================
# 7. SESSION PROFILING: K-MEANS + ISOLATION FOREST
# =============================================================================
def session_profiling(original):
    p = original[["duration_h_original", "energy_kwh", "power_kw"]].copy()
    p = p.rename(columns={"duration_h_original": "duration_h"})
    p["start_hour"] = original["start"].dt.hour.values
    p = p.replace([np.inf, -np.inf], np.nan).dropna()
    # Cap duration only for clustering stability; anomaly detection uses raw values.
    cluster_data = p.copy()
    cluster_data["duration_h"] = cluster_data["duration_h"].clip(upper=24)
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_data)
    sample_idx = np.random.default_rng(SEED).choice(
        len(X), min(MAX_CLUSTER_SAMPLE, len(X)), replace=False
    )
    rows = []
    labels_by_k = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = km.fit_predict(X)
        labels_by_k[k] = labels
        rows.append({
            "k": k,
            "silhouette": silhouette_score(X[sample_idx], labels[sample_idx]),
            "davies_bouldin": davies_bouldin_score(X, labels)
        })
    ktab = pd.DataFrame(rows)
    best_k = int(ktab.sort_values("silhouette", ascending=False).iloc[0]["k"])
    cluster_data["cluster"] = labels_by_k[best_k]
    cluster_profile = cluster_data.groupby("cluster").agg(
        duration_h=("duration_h", "mean"),
        energy_kwh=("energy_kwh", "mean"),
        power_kw=("power_kw", "mean"),
        start_hour=("start_hour", "mean"),
        sessions=("cluster", "size")
    )
    cluster_profile["share_percent"] = (
        cluster_profile.sessions / cluster_profile.sessions.sum() * 100
    )
    ktab.to_csv(RESULTS_DIR / "session_profiling_kmeans_selection.csv", index=False)
    cluster_profile.to_csv(RESULTS_DIR / "session_profiling_cluster_profiles.csv")

    # Isolation Forest is fitted on untrimmed session features.
    anomaly_features = p.copy()
    iso_scaler = StandardScaler()
    Xa = iso_scaler.fit_transform(anomaly_features)
    iso = IsolationForest(contamination=0.01, random_state=SEED)
    flags = iso.fit_predict(Xa) == -1
    anomaly_data = p.copy()
    anomaly_data["is_anomaly"] = flags
    anomaly_summary = pd.DataFrame([{
        "anomaly_count": int(flags.sum()),
        "anomaly_percent": flags.mean() * 100,
        "mean_duration_anomaly_h": anomaly_data.loc[flags, "duration_h"].mean(),
        "mean_duration_normal_h": anomaly_data.loc[~flags, "duration_h"].mean(),
        "mean_energy_anomaly_kwh": anomaly_data.loc[flags, "energy_kwh"].mean(),
        "mean_energy_normal_kwh": anomaly_data.loc[~flags, "energy_kwh"].mean()
    }])
    anomaly_summary.to_csv(RESULTS_DIR / "session_profiling_isolation_forest.csv", index=False)

    plt.figure(figsize=(6, 5))
    sm = cluster_data.sample(min(5000, len(cluster_data)), random_state=SEED)
    plt.scatter(sm.duration_h, sm.energy_kwh, c=sm.cluster, s=5)
    plt.xlabel("Duration (hours; capped at 24 for profiling)")
    plt.ylabel("Energy (kWh)")
    plt.title(f"Session profiles: K-Means (k={best_k})")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "session_profiling_clusters.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sm = anomaly_data.sample(min(8000, len(anomaly_data)), random_state=SEED)
    plt.scatter(sm.duration_h, sm.energy_kwh,
                c=np.where(sm.is_anomaly, "tab:red", "tab:blue"), s=5)
    plt.xlabel("Original duration (hours)")
    plt.ylabel("Energy (kWh)")
    plt.title("Isolation Forest anomalies")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "session_profiling_anomalies.png", dpi=200)
    plt.close()
    return ktab, cluster_profile, anomaly_summary


# =============================================================================
# 8. RECOMMENDATION EXPERIMENT + PAIRED BOOTSTRAP
# =============================================================================

def paired_bootstrap_recommendation(
    nearest_wait,
    proposed_wait,
    hour_ts,
    reps=BOOTSTRAP_REPS,
    block_len=BLOCK_LENGTH_HOURS,
    seed=SEED
):
    """Paired circular moving-block bootstrap over test hours."""

    trial_data = pd.DataFrame({
        "hour_ts": pd.to_datetime(hour_ts),
        "nearest_wait": np.asarray(nearest_wait, dtype=float),
        "proposed_wait": np.asarray(proposed_wait, dtype=float)
    })

    # Average repeated trials within each test hour.
    hourly = (
        trial_data.groupby("hour_ts")[["nearest_wait", "proposed_wait"]]
        .mean()
        .sort_index()
    )

    if hourly.empty:
        return {}

    # Preserve gaps in the hourly timeline; missing hours remain NaN.
    full_index = pd.date_range(
        hourly.index.min().floor("h"),
        hourly.index.max().floor("h"),
        freq="h"
    )
    hourly = hourly.reindex(full_index)

    nw = hourly["nearest_wait"].to_numpy(dtype=float)
    pw = hourly["proposed_wait"].to_numpy(dtype=float)
    n = len(hourly)

    valid = np.isfinite(nw) & np.isfinite(pw)
    if not valid.any():
        return {}

    point_near = float(np.nanmean(nw))
    point_prop = float(np.nanmean(pw))
    point_reduction = (
        (point_near - point_prop) / point_near * 100
        if point_near > 0 else np.nan
    )

    L = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)

    boot_near = []
    boot_prop = []
    boot_reduction = []

    for _ in range(reps):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate([
            (start + np.arange(L)) % n for start in starts
        ])[:n]

        sampled_near = nw[indices]
        sampled_prop = pw[indices]

        paired_valid = (
            np.isfinite(sampled_near) & np.isfinite(sampled_prop)
        )
        if not paired_valid.any():
            continue

        mean_near = float(np.mean(sampled_near[paired_valid]))
        mean_prop = float(np.mean(sampled_prop[paired_valid]))

        boot_near.append(mean_near)
        boot_prop.append(mean_prop)
        boot_reduction.append(
            (mean_near - mean_prop) / mean_near * 100
            if mean_near > 0 else np.nan
        )

    if not boot_near:
        return {}

    return {
        "nearest_mean_wait": point_near,
        "nearest_wait_ci_low": float(np.nanquantile(boot_near, 0.025)),
        "nearest_wait_ci_high": float(np.nanquantile(boot_near, 0.975)),
        "queue_aware_mean_wait": point_prop,
        "queue_aware_wait_ci_low": float(np.nanquantile(boot_prop, 0.025)),
        "queue_aware_wait_ci_high": float(np.nanquantile(boot_prop, 0.975)),
        "reduction_percent": point_reduction,
        "reduction_ci_low": float(np.nanquantile(boot_reduction, 0.025)),
        "reduction_ci_high": float(np.nanquantile(boot_reduction, 0.975)),
        "eligible_test_hours": int(valid.sum()),
        "bootstrap_method": "paired_circular_moving_block_by_test_hour",
        "bootstrap_block_length_hours": L,
        "bootstrap_repetitions": len(boot_near)
    }


def recommendation_trials(test_station_hours, weight, n_trials=N_RECOMMENDATION_TRIALS,
                          seed=SEED):
    # Three candidate stations at the same test hour. Distances are random by design.
    groups = [
        g for _, g in test_station_hours.groupby("hour_ts")
        if len(g) >= 3
    ]
    if not groups:
        raise ValueError("No test hours contain at least three candidate stations.")
    r = np.random.default_rng(seed)
    nearest_wait, proposed_wait = [], []
    nearest_total, proposed_total = [], []
    trial_hours = []
    for _ in range(n_trials):
        g0 = groups[int(r.integers(0, len(groups)))]
        trial_hours.append(pd.Timestamp(g0["hour_ts"].iloc[0]))
        take = r.choice(len(g0), size=3, replace=False)
        g = g0.iloc[take]
        distances = r.uniform(DISTANCE_MIN_KM, DISTANCE_MAX_KM, size=3)
        travel = distances * weight
        nearest_idx = int(np.argmin(distances))
        proposed_idx = int(np.argmin(g["pred_wait"].values + travel))
        nearest_wait.append(float(g["target_wait"].iloc[nearest_idx]))
        proposed_wait.append(float(g["target_wait"].iloc[proposed_idx]))
        nearest_total.append(float(g["target_wait"].iloc[nearest_idx] + travel[nearest_idx]))
        proposed_total.append(float(g["target_wait"].iloc[proposed_idx] + travel[proposed_idx]))
    result = paired_bootstrap_recommendation(
    nearest_wait,
    proposed_wait,
    trial_hours,
    reps=BOOTSTRAP_REPS,
    block_len=BLOCK_LENGTH_HOURS,
    seed=seed
)
    result.update({
        "distance_weight_min_per_km": weight,
        "trials": n_trials,
        "nearest_mean_total_time": np.mean(nearest_total),
        "queue_aware_mean_total_time": np.mean(proposed_total),
        "total_time_reduction_percent":
            (np.mean(nearest_total) - np.mean(proposed_total)) / np.mean(nearest_total) * 100
            if np.mean(nearest_total) > 0 else np.nan
    })
    return result


# =============================================================================
# 9. RUN ONE QUEUE EXPERIMENT FOR A DATA VERSION / CAPACITY FACTOR
# =============================================================================
def run_queue_scenario(data_version, data, factor):
    queued, cap_obs, cap_sim = replay_fcfs(data, factor)
    summary = waiting_summary(queued)
    summary.update({"dataset_version": data_version, "capacity_factor": factor})

    sh = make_station_hour_table(queued, cap_sim)
    fitted = train_validate_test_queue(sh)
    validation_table = fitted["validation_table"].copy()
    validation_table["dataset_version"] = data_version
    validation_table["capacity_factor"] = factor
    validation_table["selected_for_recommendation"] = (
        validation_table["model"] == fitted["selected_model"]
    )
    test = fitted["test"].copy()

    # Test metrics only after selection on validation.
    queue_metrics = regression_metrics(test["target_wait"], test["pred_wait"])
    queue_metrics.update({
        "model": fitted["selected_model"],
        "dataset_version": data_version,
        "capacity_factor": factor,
        "selected_model_validation_RMSE": fitted["selected_model_validation_RMSE"]
    })
    # Keep the queue formula as a benchmark, using training-period mean service time.
    train_sessions = queued[queued.hour_ts < test.hour_ts.min()]
    mean_service = train_sessions["duration_h"].mean()
    formula_pred = sakasegawa_prediction(
        test, mean_service, float(fitted["train_all"]["target_wait"].max())
    )
    formula_metrics = regression_metrics(test["target_wait"], formula_pred)
    formula_metrics.update({
        "model": "Sakasegawa-style queue approximation",
        "dataset_version": data_version,
        "capacity_factor": factor,
        "selected_model_validation_RMSE": ""
    })
    # Attach formula and learned model results.
    metrics_rows = [queue_metrics, formula_metrics]
    # Evaluate other learned models on test for a complete Table III.
    for name in ["Linear Regression", "Random Forest", "XGBoost"]:
        if name == fitted["selected_model"]:
            continue
        m = make_queue_model(name)
        m.fit(fitted["train_all"][QUEUE_FEATURES], fitted["train_all"]["target_wait"])
        p = np.maximum(0, m.predict(test[QUEUE_FEATURES]))
        rr = regression_metrics(test["target_wait"], p)
        rr.update({
            "model": name, "dataset_version": data_version,
            "capacity_factor": factor, "selected_model_validation_RMSE": ""
        })
        metrics_rows.append(rr)

    # Use validation-selected model's test predictions for recommendation.
    rec_base = test[["hour_ts", "station", "target_wait", "pred_wait"]].copy()
    rec_rows = []
    for w in DISTANCE_WEIGHTS:
        rec_rows.append(recommendation_trials(
            rec_base, w, seed=SEED + int(factor * 1000) + int(w * 10)
        ))
    rec_table = pd.DataFrame(rec_rows)
    rec_table["dataset_version"] = data_version
    rec_table["capacity_factor"] = factor

    return {
        "queued": queued,
        "summary": summary,
        "queue_metrics": pd.DataFrame(metrics_rows),
        "validation": validation_table,
        "recommendation": rec_table,
        "selected_model": fitted["selected_model"]
    }


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 78)
    print("EV CHARGING RESEARCH PIPELINE - REVISED VERSION")
    print("=" * 78)

    original, detected = load_sessions(CSV_PATH)
    original, cleaned, cleaning_stats = make_original_and_cleaned(original)

    # Dataset summary.
    ds = pd.DataFrame([{
        "sessions": len(original),
        "stations": original.station.nunique(),
        "start": original.start.min(),
        "end": original.start.max(),
        "mean_original_duration_h": original.duration_h_original.mean(),
        "mean_energy_kwh": original.energy_kwh.mean()
    }])
    ds.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    print("\nDATASET SUMMARY\n", ds.to_string(index=False))
    print("\n24-HOUR CLEANING SUMMARY\n", cleaning_stats.to_string(index=False))

    # Demand forecasting is based on observed arrivals and is not duration-cleaned.
    hourly = build_hourly_demand(original)
    forecast_table, forecast_predictions = expanding_window_forecasting(hourly)

    # Session profiling: merged clustering + anomaly analysis.
    ktab, cluster_profile, anomaly_summary = session_profiling(original)
    print("\nSESSION PROFILING - ANOMALY SUMMARY\n", anomaly_summary.to_string(index=False))

    # Original vs capped-cleaned at primary factor 0.6.
    primary_results = {}
    for label, dataset in [("original", original), ("cleaned_cap24h", cleaned)]:
        print(f"\nRunning {label} queue at capacity factor {PRIMARY_CAPACITY_FACTOR}...")
        primary_results[label] = run_queue_scenario(
            label, dataset, PRIMARY_CAPACITY_FACTOR
        )

    # Table III: original and cleaned queue prediction model metrics.
    table_iii = pd.concat(
        [primary_results["original"]["queue_metrics"],
         primary_results["cleaned_cap24h"]["queue_metrics"]],
        ignore_index=True
    )
    table_iii.to_csv(RESULTS_DIR / "table_III_queue_prediction_original_vs_cleaned.csv",
                     index=False)

    # Queue simulation summary: report both overall and conditional waiting.
    wait_summary = pd.DataFrame([
        primary_results["original"]["summary"],
        primary_results["cleaned_cap24h"]["summary"]
    ])
    wait_summary.to_csv(RESULTS_DIR / "table_queue_waiting_original_vs_cleaned.csv",
                        index=False)

    # Recommendation at alpha=2 for original and cleaned; all weights are also retained.
    rec_primary = pd.concat([
        primary_results["original"]["recommendation"],
        primary_results["cleaned_cap24h"]["recommendation"]
    ], ignore_index=True)
    rec_primary.to_csv(
        RESULTS_DIR / "table_VI_recommendation_original_vs_cleaned_all_weights.csv",
        index=False
    )

    # Capacity-factor sensitivity on cleaned (24h-capped) data.
    sensitivity_rows = []
    sensitivity_recommendations = []
    sensitivity_queue_metrics = []
    for factor in CAPACITY_FACTORS:
        if factor == PRIMARY_CAPACITY_FACTOR:
            result = primary_results["cleaned_cap24h"]
        else:
            print(f"\nRunning cleaned queue sensitivity at factor {factor}...")
            result = run_queue_scenario("cleaned_cap24h", cleaned, factor)

        s = dict(result["summary"])
        # Use validation-selected model's test metrics for the sensitivity summary.
        selected_row = result["queue_metrics"][
            result["queue_metrics"]["model"] == result["selected_model"]
        ].iloc[0]
        s.update({
            "queue_model_selected_on_validation": result["selected_model"],
            "queue_model_MAE": selected_row["MAE"],
            "queue_model_R2": selected_row["R2"]
        })
        # Primary recommendation weight requested in original paper = 2 min/km.
        rec2 = result["recommendation"][
            result["recommendation"]["distance_weight_min_per_km"] == 2.0
        ].iloc[0]
        s.update({
            "recommendation_reduction_percent": rec2["reduction_percent"],
            "recommendation_reduction_95CI_low": rec2["reduction_ci_low"],
            "recommendation_reduction_95CI_high": rec2["reduction_ci_high"]
        })
        sensitivity_rows.append(s)
        sensitivity_recommendations.append(result["recommendation"])
        sensitivity_queue_metrics.append(result["queue_metrics"])

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(RESULTS_DIR / "capacity_factor_sensitivity.csv", index=False)
    pd.concat(sensitivity_recommendations, ignore_index=True).to_csv(
        RESULTS_DIR / "recommendation_distance_weight_sensitivity_all_factors.csv",
        index=False
    )
    pd.concat(sensitivity_queue_metrics, ignore_index=True).to_csv(
        RESULTS_DIR / "queue_model_metrics_all_capacity_factors.csv",
        index=False
    )

    # Export validation model selection evidence for primary 0.6 runs.
    pd.concat([
        primary_results["original"]["validation"],
        primary_results["cleaned_cap24h"]["validation"]
    ], ignore_index=True).to_csv(
        RESULTS_DIR / "queue_validation_model_selection.csv", index=False
    )

    # Capacity sensitivity plot.
    plt.figure(figsize=(7, 4))
    plt.plot(sensitivity.capacity_factor, sensitivity.mean_wait_all_sessions_min,
             marker="o", label="Overall mean wait")
    plt.plot(sensitivity.capacity_factor, sensitivity.mean_wait_among_waiting_min,
             marker="s", label="Mean among waiting sessions")
    plt.xlabel("Capacity factor")
    plt.ylabel("Waiting time (minutes)")
    plt.title("Queue sensitivity to assumed capacity factor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "capacity_factor_wait_sensitivity.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        sensitivity.capacity_factor,
        sensitivity.recommendation_reduction_percent,
        yerr=[
            sensitivity.recommendation_reduction_percent -
            sensitivity.recommendation_reduction_95CI_low,
            sensitivity.recommendation_reduction_95CI_high -
            sensitivity.recommendation_reduction_percent
        ],
        marker="o", capsize=4
    )
    plt.xlabel("Capacity factor")
    plt.ylabel("Simulated waiting-time reduction (%)")
    plt.title("Queue-aware recommendation sensitivity (95% bootstrap CI)")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "recommendation_sensitivity_ci.png", dpi=200)
    plt.close()

    # Human-readable run metadata and methodology caveats.
    metadata = {
        "csv_path": str(CSV_PATH),
        "detected_columns": detected,
        "duration_cleaning": "Cap session duration at 24 hours; original retained for comparison",
        "capacity_factors": CAPACITY_FACTORS,
        "primary_capacity_factor": PRIMARY_CAPACITY_FACTOR,
        "queue_discipline": "FCFS",
        "recommendation_trials": N_RECOMMENDATION_TRIALS,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "recommendation_distance_method": "Random distances uniformly drawn from 0.5 to 10 km; not postcode-derived or geographic distances",
        "recommendation_weights_min_per_km": DISTANCE_WEIGHTS,
        "session_level_cleaned_export_enabled": EXPORT_SESSION_LEVEL_CLEANED,
        "queue_model_selection": "Chronological split; final 10% of pre-test period used for validation; selected model refit on all pre-test data",
        "queue_prediction_protocol": "Rolling one-hour-ahead evaluation; prior-hour observed queue outcomes are used as lag features",
        "empty_station_hour_convention": "Zero arrivals and zero mean wait; lag features computed on complete hourly station panels",
        "queue_capacity_assumption": "Station-level capacity is a scaled estimate based on maximum observed concurrency; connector compatibility is not modelled",
        "forecast_validation": "Five expanding-window chronological test periods",
        "forecast_weekly_baselines": ["Seasonal naive lag 168", "ETS seasonal period 168"],
        "license_note": "Dundee catalogue lists no licence; obtain direct confirmation before publication"
    }
    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print("Outputs saved in:", OUTPUT_DIR.resolve())
    print("Key files:")
    for name in [
        "table_dataset_cleaning_24h.csv",
        "table_II_demand_forecasting.csv",
        "forecast_xgb_vs_historical_block_bootstrap.csv",
        "table_III_queue_prediction_original_vs_cleaned.csv",
        "table_queue_waiting_original_vs_cleaned.csv",
        "capacity_factor_sensitivity.csv",
        "table_VI_recommendation_original_vs_cleaned_all_weights.csv",
        "recommendation_distance_weight_sensitivity_all_factors.csv",
        "queue_validation_model_selection.csv",
        "table_lag_correlations.csv"
    ]:
        print(" -", RESULTS_DIR / name)
    print("\nDo not copy any results into the paper until you have checked the CSV outputs.")
    print("=" * 78)


if __name__ == "__main__":
    main()
