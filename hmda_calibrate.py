"""Calibrate LendingEnv initial parameters from HMDA data.

Downloads a filtered subset of HMDA loan-application records from the CFPB
Data Browser API (https://ffiec.cfpb.gov/documentation/api/data-browser/)
and fits per-group truncated-Normal distributions over a composite
creditworthiness proxy built from income, debt-to-income ratio, and LTV.

The output JSON matches LendingConfig field names and can be passed directly
to LendingEnv:

    import json
    from lending_env import LendingEnv

    cfg = json.load(open("hmda_config.json"))
    env = LendingEnv(cfg)

API notes
---------
- The CFPB Data Browser API requires a ``states`` (or ``leis``) filter for
  the /view/csv endpoint; the /view/nationwide/csv endpoint accepts just
  ``years``. Both support additional HMDA filters (races, loan_purposes,
  actions_taken).
- The API requires at least one of: states, msamds, counties, leis.
  We default to a set of large, demographically diverse states to keep the
  download small while remaining nationally representative.
- Rows are streamed and parsed one at a time so that even large responses
  never load fully into memory.
- Year 2022 is the default (confirmed available in the API); try 2023 if
  your instance has it.

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view"

# Group A = White (historically advantaged in mortgage lending)
# Group B = Black or African American (historically disadvantaged)
RACE_A = "White"
RACE_B = "Black or African American"

# HMDA action_taken codes we include:
#   1 = Loan originated (approved & closed)
#   2 = Approved but not accepted
#   3 = Application denied
# We include all three so both approved and denied applicants shape the
# distribution, which better reflects the pool that applies.
DEFAULT_ACTIONS = "1,2,3"

# Loan purpose 1 = Home purchase (the focus of the study)
DEFAULT_LOAN_PURPOSE = "1"

# Default states: large, diverse, geographically spread.
DEFAULT_STATES = ["CA", "TX", "FL", "NY", "IL"]


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

        # iter_lines yields decoded strings without newlines; DictReader accepts
        # any string iterable, so this streams without buffering the full body.
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
    """Download up to *n_rows* filtered HMDA rows from the CFPB API.

    The API only allows 2 filter criteria (beyond years/states), so we filter
    server-side on races only and apply loan_purpose/actions_taken client-side.
    """
    params: dict[str, str] = {
        "years": str(year),
        "races": f"{RACE_A},{RACE_B}",
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

    allowed_purposes = set(loan_purpose.split(","))
    allowed_actions = set(actions.split(","))

    raw_rows = _stream_rows(url, params, n_rows * 5, timeout)
    rows: list[dict] = []
    for row in raw_rows:
        if row.get("loan_purpose", "").strip() not in allowed_purposes:
            continue
        if row.get("action_taken", "").strip() not in allowed_actions:
            continue
        rows.append(row)
        if len(rows) >= n_rows:
            break

    print(f"[fetch] received {len(rows):,} rows after client-side filtering")
    return rows


# ---------------------------------------------------------------------------
# Credit-score proxy
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


def _parse_dti(val: str) -> float | None:
    """Map HMDA DTI range string to a [0,1] score (1 = low DTI = good)."""
    v = val.strip()
    if not v or v in ("NA", "Exempt", ""):
        return None
    # Specific range tokens
    special = {
        "<20%": 0.90,
        "20%-<30%": 0.75,
        "30%-<36%": 0.62,
        "50%-60%": 0.20,
        ">60%": 0.05,
    }
    if v in special:
        return special[v]
    # Integer percentages reported as "36%", "43%", …
    try:
        pct = float(v.rstrip("%")) / 100.0
        # Linear decay: DTI 0% → 1.0, DTI 65%+ → 0.0
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
        # LTV 0% → 1.0, LTV 100%+ → 0.0
        return float(np.clip(1.0 - ltv / 100.0, 0.0, 1.0))
    except ValueError:
        return None


def compute_credit_score(row: dict) -> float | None:
    """Return a [0,1] creditworthiness proxy for one HMDA row, or None.

    Components (all in [0,1], higher = more creditworthy):
        income_score  — log-normalised applicant income
        dti_score     — inverted DTI (lower ratio → higher score)
        ltv_score     — inverted LTV (lower ratio → higher score)

    Weights: 50% income, 30% DTI, 20% LTV (LTV omitted when unavailable).
    """
    income = _parse_income(row.get("income", ""))
    dti = _parse_dti(row.get("debt_to_income_ratio", ""))

    # Both income and DTI required; LTV is optional.
    if income is None or dti is None:
        return None

    # Log-normalise income: $10k → ~0.28, $100k → ~0.57, $500k → 1.0
    income_score = float(np.clip(np.log1p(income) / np.log1p(500.0), 0.0, 1.0))

    ltv = _parse_ltv(row.get("combined_loan_to_value_ratio", ""))

    if ltv is not None:
        score = 0.50 * income_score + 0.30 * dti + 0.20 * ltv
    else:
        score = 0.625 * income_score + 0.375 * dti  # re-weight to sum to 1

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------


def fit_truncnorm(scores: np.ndarray) -> tuple[float, float]:
    """Return (mu, sigma) of the best-fit truncated Normal on [0,1].

    Uses method-of-moments: moment-matches the sample mean and std.  For the
    sample sizes we work with (thousands of applicants) this is essentially
    identical to MLE and avoids a numerical optimisation step.
    """
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

    scores_A: list[float] = []
    scores_B: list[float] = []
    n_skipped = 0

    for row in rows:
        race = row.get("derived_race", "").strip()
        score = compute_credit_score(row)
        if score is None:
            n_skipped += 1
            continue
        if race == RACE_A:
            scores_A.append(score)
        elif race == RACE_B:
            scores_B.append(score)

    arr_A = np.asarray(scores_A)
    arr_B = np.asarray(scores_B)

    print(f"\n[calibrate] skipped rows (missing income/DTI): {n_skipped:,}")
    print(f"[calibrate] Group A ({RACE_A}):              N={len(arr_A):,}")
    print(f"[calibrate] Group B ({RACE_B}): N={len(arr_B):,}")

    if arr_A.size == 0 or arr_B.size == 0:
        raise SystemExit(
            "[error] One or both groups have zero valid rows after filtering.\n"
            "Try a different year, broader state list, or larger --n-rows."
        )

    mu_A, sigma_A = fit_truncnorm(arr_A)
    mu_B, sigma_B = fit_truncnorm(arr_B)

    print(f"\n[calibrate] Group A: mu={mu_A:.4f}  sigma={sigma_A:.4f}")
    print(f"[calibrate] Group B: mu={mu_B:.4f}  sigma={sigma_B:.4f}")
    print(f"[calibrate] Raw gap (mu_A - mu_B): {mu_A - mu_B:.4f}")

    return {
        "mu_A_init": round(mu_A, 4),
        "sigma_A_init": round(sigma_A, 4),
        "N_A_init": int(len(arr_A)),
        "mu_B_init": round(mu_B, 4),
        "sigma_B_init": round(sigma_B, 4),
        "N_B_init": int(len(arr_B)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate LendingEnv initial parameters from HMDA data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="HMDA data year (2018–2023; 2022 is the confirmed-available default).",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=DEFAULT_STATES,
        metavar="ST",
        help="Two-letter state codes to include (ignored with --nationwide).",
    )
    parser.add_argument(
        "--nationwide",
        action="store_true",
        help=(
            "Use the /nationwide/csv endpoint instead of state-level filtering. "
            "Much larger download; combine with a small --n-rows."
        ),
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=50_000,
        help="Maximum number of CSV rows to stream before stopping.",
    )
    parser.add_argument(
        "--loan-purpose",
        default=DEFAULT_LOAN_PURPOSE,
        help="HMDA loan_purpose code(s): 1=home purchase, 2=home improvement, "
        "31/32=refinance.",
    )
    parser.add_argument(
        "--actions",
        default=DEFAULT_ACTIONS,
        help="HMDA action_taken code(s) to include.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--out",
        default="hmda_config.json",
        help="Output path for the calibrated config JSON.",
    )
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
