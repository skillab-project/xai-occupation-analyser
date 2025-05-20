import pytest
import pandas as pd
import os
from fetch_specific_occupation4d_skills import fetch_and_store_specific_occupation_skills

def test_fetch_specific_skills_success(mocker):
    occupation_ids = ["http://data.europa.eu/esco/isco/C2511"]
    jobs_response = {
        "items": [
            {"id": "job1", "upload_date": "2024-12-01", "skills": [301, 302]},
            {"id": "job2", "upload_date": "2024-12-02", "skills": [302]}
        ],
        "count": 2
    }
    skills_response = {"items": [{"id": 301, "label": "SQL"}, {"id": 302, "label": "Docker"}], "count": 2}

    def side_effect(url, data, headers=None):
        if "jobs" in url:
            return mocker.Mock(status_code=200, json=lambda: jobs_response)
        elif "skills" in url:
            return mocker.Mock(status_code=200, json=lambda: skills_response)
        raise ValueError("Unknown URL")

    mocker.patch("fetch_specific_occupation4d_skills.get_token", return_value="dummy_token")
    mocker.patch("fetch_specific_occupation4d_skills.requests.post", side_effect=side_effect)

    result_path = fetch_and_store_specific_occupation_skills(occupation_ids)
    assert os.path.exists(result_path)
    df = pd.read_csv(result_path)
    assert "SQL" in df.columns
    assert "Docker" in df.columns
    os.remove(result_path)
