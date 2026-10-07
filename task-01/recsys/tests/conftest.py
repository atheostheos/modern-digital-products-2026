from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).parent
LEVELS = ("unit", "integration", "e2e")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test with its level, taken from its top-level tests/ subdirectory."""
    for item in items:
        level = Path(item.path).relative_to(TESTS_ROOT).parts[0]
        if level in LEVELS:
            item.add_marker(getattr(pytest.mark, level))
