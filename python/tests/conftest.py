"""Pytest configuration for GradFlow tests."""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample test data."""
    return {"test": "data"}
