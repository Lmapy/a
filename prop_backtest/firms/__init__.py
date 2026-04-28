from .base import FirmRules, AccountTier, DrawdownType
from .topstep import TOPSTEP_RULES
from .myfundedfutures import MFF_RULES
from .lucid import LUCID_RULES

FIRM_REGISTRY: dict[str, FirmRules] = {
    "topstep": TOPSTEP_RULES,
    "myfundedfutures": MFF_RULES,
    "mff": MFF_RULES,
    "lucid": LUCID_RULES,
}


def get_firm(name: str) -> FirmRules:
    """Look up firm rules by name (case-insensitive)."""
    key = name.lower().replace(" ", "").replace("-", "")
    if key not in FIRM_REGISTRY:
        available = ", ".join(FIRM_REGISTRY.keys())
        raise ValueError(f"Unknown firm '{name}'. Available: {available}")
    return FIRM_REGISTRY[key]


__all__ = [
    "FirmRules", "AccountTier", "DrawdownType",
    "TOPSTEP_RULES", "MFF_RULES", "LUCID_RULES",
    "FIRM_REGISTRY", "get_firm",
]
