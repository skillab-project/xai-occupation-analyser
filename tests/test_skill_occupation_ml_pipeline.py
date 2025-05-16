import pandas as pd
import os
from skill_occupation_ml_pipeline import train_and_evaluate_model

def test_train_and_evaluate_model(tmp_path):
    # 3 classes, 50 samples each = 150 samples total
    num_classes = 3
    samples_per_class = 50
    records = []

    for i in range(num_classes):
        label = f"C{i+1}"
        for j in range(samples_per_class):
            records.append({
                "job_id": len(records) + 1,
                "upload_date": "2024-12-01",
                "occupation4d": label,
                "Python": j % 2,
                "ML": (j + i) % 2
            })

    df = pd.DataFrame(records)
    file_path = tmp_path / "test_skills.csv"
    df.to_csv(file_path, index=False)

    result_file = train_and_evaluate_model(str(file_path), min_category_samples=30, test_size=0.2)

    assert os.path.exists(result_file)
    metrics_df = pd.read_csv(result_file)
    assert "AUC" in metrics_df.columns
    assert not metrics_df.empty
    os.remove(result_file)
