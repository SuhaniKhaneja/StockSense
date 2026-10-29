import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Point the API at a small synthetic forecast so tests do not depend on
    whatever the pipeline last wrote to artifacts/."""
    dates = pd.date_range("2024-01-01", periods=14)
    rows = []
    for sku in ("SKU-001", "SKU-002"):
        p = np.linspace(5, 10, 14)
        rows.append(pd.DataFrame({"date": dates, "sku": sku, "pred": p, "lo": p * 0.5, "hi": p * 1.5}))
    pd.concat(rows).to_csv(tmp_path / "forecast.csv", index=False)
    monkeypatch.setattr(api, "ART", tmp_path)
    return TestClient(api.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["forecasts_ready"] is True


def test_forecast_ok_and_unknown_sku(client):
    r = client.get("/forecast/SKU-001")
    assert r.status_code == 200 and len(r.json()["forecast"]) == 14
    assert client.get("/forecast/NOPE").status_code == 404


def test_recommendations_validates_input(client):
    ok = client.get("/recommendations?lead_time=5")
    assert ok.status_code == 200 and {"sku", "action", "order_qty"} <= set(ok.json()[0])
    assert client.get("/recommendations?lead_time=99").status_code == 422


def test_503_when_no_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ART", tmp_path)
    assert TestClient(api.app).get("/forecast/SKU-001").status_code == 503
