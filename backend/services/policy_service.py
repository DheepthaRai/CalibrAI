"""
Policy recommendation engine.
Derives the optimal safety threshold (1–5) from three inputs:
  - industry
  - cost_per_violation ($)
  - weekly_volume (queries/week)
"""

from __future__ import annotations

# ============================================================================
# INDUSTRY RISK PROFILES
# ============================================================================

INDUSTRY_BASE_LEVEL = {
    "Banking": 2,           # Strict  — high regulatory exposure
    "Healthcare": 1,        # Very Strict — HIPAA, patient safety
    "Insurance": 2,         # Strict — fraud risk
    "E-commerce": 3,        # Balanced — volume-driven, moderate risk
    "Customer Service": 3,  # Balanced — utility matters most
}

INDUSTRY_RISK_LABELS = {
    "Banking": "High regulatory exposure (BSA / AML compliance)",
    "Healthcare": "Critical — HIPAA & patient safety obligations",
    "Insurance": "High fraud risk — regulatory and financial exposure",
    "E-commerce": "Moderate — reputational and payment-fraud risk",
    "Customer Service": "Low-moderate — brand and user-experience risk",
}

LEVEL_NAMES = {
    1: "Very Strict",
    2: "Strict",
    3: "Balanced",
    4: "Permissive",
    5: "Very Permissive",
}

LEVEL_DESCRIPTIONS = {
    1: "Block anything remotely suspicious. Maximum safety, highest user friction.",
    2: "Block likely-harmful requests and sensitive-data exposure. Err on side of caution.",
    3: "Block clearly harmful requests. Allow legitimate use with context. Recommended default.",
    4: "Only block obvious attacks. High utility, accepts some residual risk.",
    5: "Only refuse explicitly illegal content. Maximum utility, minimum filtering.",
}


# ============================================================================
# CORE RECOMMENDATION LOGIC
# ============================================================================

def recommend_threshold(
    industry: str,
    cost_per_violation: float,
    weekly_volume: int,
) -> dict:
    """
    Returns the recommended safety level (1–5) with explanation and cost estimates.
    """
    base = INDUSTRY_BASE_LEVEL.get(industry, 3)

    # --- Adjust by cost-per-violation ---
    # Higher financial exposure → tighten the threshold
    if cost_per_violation >= 10_000:
        base = min(base, 1)
    elif cost_per_violation >= 5_000:
        base = min(base, 2)
    elif cost_per_violation >= 1_000:
        base = min(base, 3)

    # --- Adjust by weekly volume ---
    # At very high volume, false positives become the dominant cost driver,
    # so we relax the floor slightly to protect user experience.
    if weekly_volume >= 100_000:
        base = max(base, 3)
    elif weekly_volume >= 50_000:
        base = max(base, 2)

    recommended_level = base

    # --- Cost estimates (rough model) ---
    # Estimated attack rate: 5% of queries
    # Estimated FP rate per level: 20% at L1, 12% at L2, 6% at L3, 2% at L4, 0.5% at L5
    fp_rates = {1: 0.20, 2: 0.12, 3: 0.06, 4: 0.02, 5: 0.005}
    attack_rates = {1: 0.01, 2: 0.02, 3: 0.05, 4: 0.12, 5: 0.20}  # missed attacks

    weekly_attacks = weekly_volume * 0.05  # ~5% of queries are attacks
    weekly_legit = weekly_volume * 0.95

    fp_rate = fp_rates.get(recommended_level, 0.06)
    miss_rate = attack_rates.get(recommended_level, 0.05)

    weekly_fp_cost = weekly_legit * fp_rate * (cost_per_violation * 0.02)  # FP = 2% of violation cost
    weekly_attack_cost = weekly_attacks * miss_rate * cost_per_violation

    explanation = _build_explanation(
        industry, cost_per_violation, weekly_volume, recommended_level,
        weekly_fp_cost, weekly_attack_cost,
    )

    return {
        "recommended_level": recommended_level,
        "level_name": LEVEL_NAMES[recommended_level],
        "level_description": LEVEL_DESCRIPTIONS[recommended_level],
        "explanation": explanation,
        "industry_risk_profile": INDUSTRY_RISK_LABELS.get(industry, ""),
        "weekly_fp_cost_estimate": round(weekly_fp_cost, 2),
        "weekly_attack_cost_estimate": round(weekly_attack_cost, 2),
        "inputs": {
            "industry": industry,
            "cost_per_violation": cost_per_violation,
            "weekly_volume": weekly_volume,
        },
    }


def _build_explanation(
    industry: str,
    cost: float,
    volume: int,
    level: int,
    fp_cost: float,
    attack_cost: float,
) -> str:
    name = LEVEL_NAMES[level]
    parts = [
        f"For a {industry} deployment processing {volume:,} queries/week "
        f"with a ${cost:,.0f} cost per compliance violation:",
    ]

    if level == 1:
        parts.append(
            f"Level 1 (Very Strict) is required. The combination of high regulatory exposure "
            f"and significant per-violation cost means even rare breaches are unacceptably expensive. "
            f"Estimated missed-attack cost: ${attack_cost:,.0f}/week."
        )
    elif level == 2:
        parts.append(
            f"Level 2 (Strict) provides the right balance. Your regulatory environment demands "
            f"strong filtering, but extreme strictness would generate ~${fp_cost:,.0f}/week in "
            f"user-friction cost. Level 2 blocks the vast majority of threats while keeping FPs manageable."
        )
    elif level == 3:
        parts.append(
            f"Level 3 (Balanced) is the optimal default. At your volume and cost profile, "
            f"the friction cost of over-blocking (~${fp_cost:,.0f}/week) outweighs the marginal "
            f"security gain from stricter levels. Level 3 catches clear attacks while remaining "
            f"highly usable."
        )
    elif level == 4:
        parts.append(
            f"Level 4 (Permissive) maximises utility for your use case. With a lower cost-per-violation "
            f"and high query volume, the business cost of false positives (~${fp_cost:,.0f}/week) "
            f"dominates. Level 4 only blocks obvious attacks, minimising user friction."
        )
    else:
        parts.append(
            f"Level 5 (Very Permissive) is appropriate given your very high volume and low "
            f"per-incident cost. Only explicitly illegal content is blocked."
        )

    return " ".join(parts)
