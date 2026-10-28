from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_forecast_ok_and_unknown_sku():
    r = client.get("/forecast/SKU-001")
    assert r.status_code == 200 and len(r.json()["forecast"]) == 14
    assert client.get("/forecast/NOPE").status_code == 404


def test_recommendations_validates_input():
    ok = client.get("/recommendations?lead_time=5")
    assert ok.status_code == 200 and {"sku", "action", "order_qty"} <= set(ok.json()[0])
    assert client.get("/recommendations?lead_time=99").status_code == 422
