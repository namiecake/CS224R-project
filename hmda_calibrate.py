"""Calibrate LendingEnv initial parameters from HMDA data.

Downloads a filtered subset of HMDA loan-application records from the CFPB
Data Browser API and fits per-group truncated-Normal distributions over a
composite creditworthiness proxy, bucketed into six Race × Income groups.

Usage
-----
    python hmda_calibrate.py                          # defaults
    python hmda_calibrate.py --states CA TX FL NY IL --year 2022 --n-rows 50000
    python hmda_calibrate.py --nationwide --year 2022 --n-rows 100000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Iterator

import numpy as np
import requests

from constants import (
    GROUP_ORDER,
    HMDA_ETHNICITY_HISPANIC,
    HMDA_RACE_BLACK,
    HMDA_RACE_WHITE,
)
from lending_env import weighted_variance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view"

DEFAULT_ACTIONS = "1,2,3"
DEFAULT_LOAN_PURPOSE = "1"
DEFAULT_STATES = ["CA", "TX", "FL", "NY", "IL"]

MIN_GROUP_N = 50


# ---------------------------------------------------------------------------
# API fetch — streaming
# ---------------------------------------------------------------------------


_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; hmda-calibrate/1.0)"}


def _stream_rows(url: str, params: dict, n_rows: int, timeout: int) -> Iterator[dict]:
    """Yield up to *n_rows* parsed CSV rows from a streaming CFPB API response."""
    with requests.get(url, params=params, headers=_HEADERS, stream=True, timeout=timeout) as r:
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            print(
                f"[error] API returned {r.status_code}: {r.text[:400]}",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        lines = r.iter_lines(decode_unicode=True)
        reader = csv.DictReader(lines)
        count = 0
        for row in reader:
            yield row
            count += 1
            if count >= n_rows:
                break


def fetch_rows(
    *,
    year: int,
    states: list[str] | None,
    nationwide: bool,
    n_rows: int,
    loan_purpose: str,
    actions: str,
    timeout: int,
) -> list[dict]:
    """Download up to *n_rows* filtered HMDA rows from the CFPB API."""
    params: dict[str, str] = {
        "years": str(year),
        "races": f"{HMDA_RACE_WHITE},{HMDA_RACE_BLACK}",
    }

    if nationwide:
        url = f"{API_BASE}/nationwide/csv"
    else:
        if not states:
            states = DEFAULT_STATES
        params["states"] = ",".join(s.upper() for s in states)
        url = f"{API_BASE}/csv"

    print(f"[fetch] GET {url}")
    print(f"        params: {params}")
    print(f"        streaming up to {n_rows:,} rows …")
    print(f"        client-side filters: loan_purpose={loan_purpose!r}, actions_taken={actions!r}")
    print(
        "        client-side ethnicity filter: drop Hispanic White applicants"
    )

    allowed_purposes = set(loan_purpose.split(","))
    allowed_actions = set(actions.split(","))

    raw_rows = _stream_rows(url, params, n_rows * 5, timeout)
    rows: list[dict] = []
    n_ethnicity_dropped = 0
    for row in raw_rows:
        if row.get("loan_purpose", "").strip() not in allowed_purposes:
            continue
        if row.get("action_taken", "").strip() not in allowed_actions:
            continue
        race = row.get("derived_race", "").strip()
        ethnicity = row.get("derived_ethnicity", "").strip()
        if race == HMDA_RACE_WHITE and ethnicity == HMDA_ETHNICITY_HISPANIC:
            n_ethnicity_dropped += 1
            continue
        rows.append(row)
        if len(rows) >= n_rows:
            break

    print(
        f"[fetch] received {len(rows):,} rows after client-side filtering "
        f"({n_ethnicity_dropped:,} Hispanic White rows dropped)"
    )
    return rows


# ---------------------------------------------------------------------------
# Credit-score proxy and group assignment
# ---------------------------------------------------------------------------


def _parse_income(val: str) -> float | None:
    """Return applicant income in $thousands, or None if unavailable."""
    v = val.strip()
    if not v or v in ("NA", "Exempt", "9999", ""):
        return None
    try:
        income = float(v)
        return income if income > 0 else None
    except ValueError:
        return None


def _parse_ami(val: str) -> float | None:
    """Return area median family income in dollars, or None if unavailable."""
    v = val.strip()
    if not v or v in ("NA", "Exempt", ""):
        return None
    try:
        ami = float(v)
        return ami if ami > 0 else None
    except ValueError:
        return None


def _parse_dti(val: str) -> float | None:
    """Map HMDA DTI range string to a [0,1] score (1 = low DTI = good)."""
    v = val.strip()
    if not v or v in ("NA", "Exempt", ""):
        return None
    special = {
        "<20%": 0.90,
        "20%-<30%": 0.75,
        "30%-<36%": 0.62,
        "50%-60%": 0.20,
        ">60%": 0.05,
    }
    if v in special:
        return special[v]
    try:
        pct = float(v.rstrip("%")) / 100.0
        return float(np.clip(1.0 - pct / 0.65, 0.0, 1.0))
    except ValueError:
        return None


def _parse_ltv(val: str) -> float | None:
    """Map combined LTV ratio (%) to a [0,1] score (1 = low LTV = good)."""
    v = val.strip()
    if not v or v in ("NA", "Exempt", ""):
        return None
    try:
        ltv = float(v)
        return float(np.clip(1.0 - ltv / 100.0, 0.0, 1.0))
    except ValueError:
        return None


def compute_credit_score(row: dict) -> float | None:
    """Return a [0,1] creditworthiness proxy for one HMDA row, or None."""
    income = _parse_income(row.get("income", ""))
    dti = _parse_dti(row.get("debt_to_income_ratio", ""))
    if income is None or dti is None:
        return None

    income_score = float(np.clip(np.log1p(income) / np.log1p(500.0), 0.0, 1.0))
    ltv = _parse_ltv(row.get("combined_loan_to_value_ratio", ""))

    if ltv is not None:
        score = 0.50 * income_score + 0.30 * dti + 0.20 * ltv
    else:
        score = 0.625 * income_score + 0.375 * dti

    return float(np.clip(score, 0.0, 1.0))


def compute_income_tier(row: dict) -> str | None:
    """Return ``Low``, ``Middle``, or ``Upper`` from income-to-AMI ratio."""
    income = _parse_income(row.get("income", ""))
    ami = _parse_ami(row.get("ffiec_msa_md_median_family_income", ""))
    if income is None or ami is None:
        return None
    ratio = (income * 1000.0) / ami
    if ratio < 0.80:
        return "Low"
    if ratio < 1.20:
        return "Middle"
    return "Upper"


def assign_group(row: dict) -> str | None:
    """Return a canonical group name or ``None`` if the row cannot be assigned."""
    race = row.get("derived_race", "").strip()
    ethnicity = row.get("derived_ethnicity", "").strip()

    if race == HMDA_RACE_WHITE:
        if ethnicity == HMDA_ETHNICITY_HISPANIC:
            return None
        race_label = "White"
    elif race == HMDA_RACE_BLACK:
        race_label = "Black"
    else:
        return None

    tier = compute_income_tier(row)
    if tier is None:
        return None

    name = f"{race_label}_{tier}"
    return name if name in GROUP_ORDER else None


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------


def fit_truncnorm(scores: np.ndarray) -> tuple[float, float]:
    """Return (mu, sigma) of the best-fit truncated Normal on [0,1]."""
    if scores.size == 0:
        raise ValueError("Cannot fit: no valid scores for this group.")
    mu = float(scores.mean())
    sigma = float(max(scores.std(), 0.01))
    return mu, sigma


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def calibrate(
    *,
    year: int,
    states: list[str] | None,
    nationwide: bool,
    n_rows: int,
    loan_purpose: str,
    actions: str,
    timeout: int,
) -> dict:
    """Fetch HMDA data, fit distributions, return a LendingEnv config dict."""
    rows = fetch_rows(
        year=year,
        states=states,
        nationwide=nationwide,
        n_rows=n_rows,
        loan_purpose=loan_purpose,
        actions=actions,
        timeout=timeout,
    )

    scores_by_group: dict[str, list[float]] = {name: [] for name in GROUP_ORDER}
    n_skipped_score = 0
    n_skipped_group = 0

    for row in rows:
        group = assign_group(row)
        score = compute_credit_score(row)
        if score is None:
            n_skipped_score += 1
            continue
        if group is None:
            n_skipped_group += 1
            continue
        scores_by_group[group].append(score)

    print(f"\n[calibrate] skipped rows (missing income/DTI): {n_skipped_score:,}")
    print(f"[calibrate] skipped rows (unassigned group):  {n_skipped_group:,}")

    print(f"\n{'Group':<16} {'N':>8} {'mean':>8} {'std':>8}")
    print("-" * 44)

    group_configs: list[dict] = []
    mus: list[float] = []
    Ns: list[float] = []

    for name in GROUP_ORDER:
        race, tier = name.split("_", 1)
        arr = np.asarray(scores_by_group[name])
        n = int(arr.size)

        if n == 0:
            print(f"{name:<16} {n:>8} {'—':>8} {'—':>8}")
            mu, sigma = 0.0, 0.01
        else:
            mu, sigma = fit_truncnorm(arr)
            print(f"{name:<16} {n:>8} {mu:>8.4f} {sigma:>8.4f}")
            mus.append(mu)
            Ns.append(float(n))

        group_configs.append(
            {
                "name": name,
                "race": race,
                "income_tier": tier,
                "mu_init": round(mu, 4),
                "sigma_init": round(sigma, 4),
                "N_init": n,
            }
        )

    if mus:
        weights = np.asarray(Ns, dtype=np.float64)
        means = np.asarray(mus, dtype=np.float64)
        overall_mean = float(np.sum(weights * means) / weights.sum())
        wvar = weighted_variance(means, weights)
        print(f"\n[calibrate] Population-weighted overall mean: {overall_mean:.4f}")
        print(f"[calibrate] Weighted variance of group means:   {wvar:.6f}")

    small_groups = [g["name"] for g in group_configs if g["N_init"] < MIN_GROUP_N]
    if small_groups:
        print(
            f"\n*** WARNING: {len(small_groups)} group(s) have N < {MIN_GROUP_N}: "
            f"{', '.join(small_groups)}. "
            "Distribution estimates may be unreliable. "
            "Try a larger --n-rows or different state filter."
        )

    if all(g["N_init"] == 0 for g in group_configs):
        raise SystemExit(
            "[error] All groups have zero valid rows after filtering.\n"
            "Try a different year, broader state list, or larger --n-rows."
        )

    return {"groups": group_configs}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate LendingEnv initial parameters from HMDA data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument(
        "--states",
        nargs="+",
        default=DEFAULT_STATES,
        metavar="ST",
        help="Two-letter state codes (ignored with --nationwide).",
    )
    parser.add_argument("--nationwide", action="store_true")
    parser.add_argument("--n-rows", type=int, default=50_000)
    parser.add_argument("--loan-purpose", default=DEFAULT_LOAN_PURPOSE)
    parser.add_argument("--actions", default=DEFAULT_ACTIONS)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", default="hmda_config.json")
    args = parser.parse_args()

    cfg = calibrate(
        year=args.year,
        states=args.states,
        nationwide=args.nationwide,
        n_rows=args.n_rows,
        loan_purpose=args.loan_purpose,
        actions=args.actions,
        timeout=args.timeout,
    )

    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[done] Calibrated config saved to '{args.out}':")
    print(json.dumps(cfg, indent=2))
    print(
        "\nTo use in LendingEnv:\n"
        "    import json\n"
        "    from lending_env import LendingEnv\n"
        f"    env = LendingEnv(json.load(open('{args.out}')))\n"
    )


if __name__ == "__main__":
    main()
