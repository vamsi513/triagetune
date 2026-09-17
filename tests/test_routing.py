from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import load_categories
from src.routing import build_routes


APPROVED = set(
    load_categories(Path(__file__).resolve().parents[1] / "reports" / "lora_adapter_test.json")
)


def test_every_category_has_one_provisional_route():
    routes = build_routes(APPROVED)
    assert len(routes) == 77
    assert set(routes) == APPROVED
    assert {route.priority for route in routes.values()} == {"urgent", "standard", "low"}
    assert len({route.team for route in routes.values()}) == 7
    assert routes["compromised_card"].priority == "urgent"
    assert routes["pin_blocked"].priority == "standard"
    assert routes["exchange_rate"].priority == "low"


def test_changed_category_set_cannot_inherit_a_route():
    with pytest.raises(ValueError, match="exactly once"):
        build_routes(APPROVED - {"pin_blocked"})
    with pytest.raises(ValueError, match="exactly once"):
        build_routes(APPROVED | {"invented"})
