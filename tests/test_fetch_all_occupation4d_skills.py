import pytest
import pandas as pd
import os
from fetch_all_occupation4d_skills import fetch_and_store_all_occupation_skills

def test_fetch_all_skills_success(mocker):
    occupation_response = {"items": [{"id": 101}, {"id": 102}], "count": 2}
    jobs_response = {
        "items": [
            {"id": "job1", "upload_date": "2024-12-01", "skills": [201, 202]},
            {"id": "job2", "upload_date": "2024-12-02", "skills": [202]}
        ],
        "count": 2
    }
    skills_response = {"items": [{"id": 201, "label": "Python"}, {"id": 202, "label": "ML"}], "count": 2}

    def side_effect(url, data, headers=None):
        if "occupations" in url:
            return mocker.Mock(status_code=200, json=lambda: occupation_response)
        elif "jobs" in url:
            return mocker.Mock(status_code=200, json=lambda: jobs_response)
        elif "skills" in url:
            return mocker.Mock(status_code=200, json=lambda: skills_response)
        raise ValueError("Unknown URL")

    mocker.patch("fetch_all_occupation4d_skills.get_token", return_value="dummy_token")
    mocker.patch("fetch_all_occupation4d_skills.requests.post", side_effect=side_effect)

    result_path = fetch_and_store_all_occupation_skills()
    assert os.path.exists(result_path)
    df = pd.read_csv(result_path)
    assert "Python" in df.columns
    assert "ML" in df.columns
    os.remove(result_path)
