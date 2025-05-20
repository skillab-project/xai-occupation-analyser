# xai-occupation-analyser

## Overview
The backend of the XAI Occupation Analyser, a ML framework augmented with XAI mechanisms for the identification, prioritization, and interpretation of the importance of the most relevant skills for specific occupation and job roles.

This FastAPI application provides endpoints to:
1. Fetch and store all occupation skills from an API.
2. Fetch and store specific occupation skills based on provided occupation IDs.
3. Train and evaluate a machine learning model using a dataset.
4. Analyze feature importance of an occupation.
5. Retrieve SHAP plots for feature analysis.

## Installation

### Prerequisites
Ensure you have **Python 3.8+** installed.

### Install Dependencies
```sh
pip install fastapi uvicorn pandas numpy scikit-learn matplotlib shap requests pydantic python-dotenv
```

## Running the API
To start the API server, run:
```sh
uvicorn main:app --reload
```

The API will be available at:
```
http://127.0.0.1:8000
```

## Running with Docker
Build the Docker image:
```sh
docker build -t xai-occupation-analyser .
```

Run the container:
```
docker run -p 8000:8000 xai-occupation-analyser
```

The API will be available at:
```
http://127.0.0.1:8000
```

## API Endpoints

### 1. Fetch All Occupation Skills
**Endpoint:**
```http
POST /fetch_all_occupation_skills
```
**Request Body:**
```json
{
  "min_level": 3,
  "max_level": 3,
  "min_skill_level": 4,
  "max_skill_level": 4,
  "sources": ["kariera.gr", "lesjeudis"]
}
```
**Response:**
```json
{ "task_id": "123e4567-e89b-12d3-a456-426614174000", "message": "Fetching started. Check progress using /task_status/{task_id}" }
```

### 2. Fetch Specific Occupation Skills
**Endpoint:**
```http
POST /fetch_specific_occupation_skills
```
**Request Body:**
```json
{
  "occupation_ids": ["http://data.europa.eu/esco/isco/C2511", "http://data.europa.eu/esco/isco/C2512", "http://data.europa.eu/esco/isco/C2513", "http://data.europa.eu/esco/isco/C2514", "http://data.europa.eu/esco/isco/C2519"],
  "min_skill_level": 4,
  "max_skill_level": 4,
  "sources": ["kariera.gr", "lesjeudis"]
}
```
**Response:**
```json
{ "task_id": "123e4567-e89b-12d3-a456-426614174000", "message": "Fetching started. Check progress using /task_status/{task_id}" }
```

### 3. Train and Evaluate Model
**Endpoint:**
```http
POST /train_and_evaluate
```
**Request Body:**
```json
{
  "file_path": "datasets/all_occupation4d_skills.csv",
  "min_category_samples": 50,
  "test_size": 0.1
}
```
**Response:**
```json
{ "task_id": "123e4567-e89b-12d3-a456-426614174000", "message": "Model training started. Check progress using /task_status/{task_id}" }
```

### 4. Track Task Status
**Endpoint:**
```http
GET /task_status/{task_id}
```
**Example Request:**
```http
GET /task_status/123e4567-e89b-12d3-a456-426614174000
```
**Example Responses:**
- **In Progress:**
  ```json
  {
    "status": "in_progress",
    "file_path": ""
  }
  ```
- **Completed:**
  ```json
  {
    "status": "completed",
    "file_path": "datasets/all_occupation4d_skills.csv"
  }
  ```
- **Failed:**
  ```json
  {
    "status": "failed",
    "error": "Some error message"
  }
  ```

### 5. Analyze Occupation Feature Importance
**Endpoint:**
```http
POST /analyze_occupation
```
**Request Body:**
```json
{
  "occupation_id_url": "http://data.europa.eu/esco/isco/C2511"
}
```
**Response:**
```json
{
  "message": "Analysis completed for C2511",
  "top_features": [ { "Feature": "skill_x", "Importance": 0.85 } ],
  "bar_plot": "/plots/shap_bar_plot_C2511.png",
  "dot_plot": "/plots/shap_dot_plot_C2511.png"
}
```

### 6. Retrieve SHAP Plot
**Endpoint:**
```http
GET /plots/{plot_name}
```
**Example Request:**
```http
GET /plots/shap_bar_plot_C2511.png
```

## Running Unit Tests
This project uses **pytest**. Tests are located under the `tests/` folder.

### Install test dependencies:
```sh
pip install pytest pytest-mock
```

### Run tests:
```sh
pytest
```
Make sure to run from the project root (where `main.py` lives), not from inside `tests/`.

## Project Structure
```
├── main.py                          # FastAPI app
├── fetch_all_occupation4d_skills.py
├── fetch_specific_occupation4d_skills.py
├── skill_occupation_ml_pipeline.py
├── tests/
│   ├── test_fetch_all_occupation4d_skills.py
│   ├── test_fetch_specific_occupation4d_skills.py
│   ├── test_main_api.py
│   └── test_skill_occupation_ml_pipeline.py
├── models/                          # Trained models (auto-created)
├── plots/                           # SHAP visualizations
├── feature_importance/              # CSVs with top skill importances
├── datasets/                        # Raw data
├── Dockerfile
└── README.md
```

## Contributing
Feel free to contribute by submitting issues or pull requests.

## License
TBA
