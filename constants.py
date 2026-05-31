"""Shared constants for the 6-group lending simulator (Race × Income)."""

from __future__ import annotations

# Canonical group order: state vector, action vector, plots, config JSON.
GROUP_ORDER: tuple[str, ...] = (
    "White_Low",
    "White_Middle",
    "White_Upper",
    "Black_Low",
    "Black_Middle",
    "Black_Upper",
)

RACES: tuple[str, ...] = ("White", "Black")
INCOME_TIERS: tuple[str, ...] = ("Low", "Middle", "Upper")

# HMDA derived_race values used in API filters and assignment.
HMDA_RACE_WHITE = "White"
HMDA_RACE_BLACK = "Black or African American"
HMDA_ETHNICITY_HISPANIC = "Hispanic or Latino"

# Plot styling: color by race, linestyle by income tier.
RACE_COLORS: dict[str, str] = {
    "White": "C0",
    "Black": "C1",
}
TIER_LINESTYLES: dict[str, str] = {
    "Low": "-",
    "Middle": "--",
    "Upper": ":",
}


def parse_group_name(name: str) -> tuple[str, str]:
    """Return ``(race, income_tier)`` for a canonical group name."""
    race, tier = name.split("_", 1)
    return race, tier
