from __future__ import annotations

import pytest


@pytest.mark.integration
def test_legacy_lightning_runtime_has_pkg_resources() -> None:
    import pkg_resources  # noqa: F401
    pytest.importorskip("pytorch_lightning")
