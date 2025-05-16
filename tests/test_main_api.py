import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_task_status_not_found():
    response = client.get("/task_status/invalid-id")
    assert response.status_code == 404

def test_invalid_analyze_occupation_url():
    response = client.post("/analyze_occupation", json={"occupation_id_url": "not-a-url"})
    assert response.status_code == 400
