# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:23:34 2025

@author: Dimitris Tsoukalas (tsoukj)
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urlparse
import pandas as pd
import re
import uuid
import time  # Simulating a long-running process

from fetch_all_occupation4d_skills import fetch_and_store_all_occupation_skills
from fetch_specific_occupation4d_skills import fetch_and_store_specific_occupation_skills
from skill_occupation_ml_pipeline import train_and_evaluate_model

app = FastAPI()

# Define folders
output_folder_features = Path("feature_importance")
plots_folder = Path("plots")
datasets_folder = Path("datasets")

# Ensure necessary folders exist
for folder in [output_folder_features, plots_folder, datasets_folder]:
    folder.mkdir(exist_ok=True)

# Dictionary to store task statuses
task_status: Dict[str, Dict[str, str]] = {}

# Define input schema for the API
class OccupationRequest(BaseModel):
    occupation_id_url: str

class FetchAllOccupationSkillsRequest(BaseModel):
    min_level: Optional[int] = 3
    max_level: Optional[int] = 3
    min_skill_level: Optional[int] = 4
    max_skill_level: Optional[int] = 4
    sources: Optional[List[str]] = None

class FetchSpecificOccupationSkillsRequest(BaseModel):
    occupation_ids: List[str]
    min_skill_level: Optional[int] = 4
    max_skill_level: Optional[int] = 4
    sources: Optional[List[str]] = None

class TrainModelRequest(BaseModel):
    file_path: Optional[str] = "datasets/all_occupation4d_skills.csv"
    min_category_samples: Optional[int] = 50
    test_size: Optional[float] = 0.1

### **API ENDPOINTS** ###

@app.post("/fetch_all_occupation_skills")
async def fetch_all_occupation_skills_api(request: FetchAllOccupationSkillsRequest, background_tasks: BackgroundTasks):
    """Fetches and stores all occupation skills."""
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"status": "in_progress", "file_path": ""}
    background_tasks.add_task(fetch_and_store_all_occupation_skills_with_status, request, task_id)
    return {"task_id": task_id, "message": f"Fetching started. Check progress using /task_status/{task_id}"}

def fetch_and_store_all_occupation_skills_with_status(request: FetchAllOccupationSkillsRequest, task_id: str):
    try:
        file_path = fetch_and_store_all_occupation_skills(request.min_level, request.max_level, request.min_skill_level, request.max_skill_level, sources=request.sources)
        task_status[task_id] = {"status": "completed", "file_path": file_path}
    except Exception as e:
        task_status[task_id] = {"status": "failed", "error": str(e)}

@app.post("/fetch_specific_occupation_skills")
async def fetch_specific_occupation_skills_api(request: FetchSpecificOccupationSkillsRequest, background_tasks: BackgroundTasks):
    """Fetches and stores specific occupation skills."""
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"status": "in_progress", "file_path": ""}
    background_tasks.add_task(fetch_and_store_specific_occupation_skills_with_status, request, task_id)
    return {"task_id": task_id, "message": f"Fetching started. Check progress using /task_status/{task_id}"}

def fetch_and_store_specific_occupation_skills_with_status(request: FetchSpecificOccupationSkillsRequest, task_id: str):
    try:
        file_path = fetch_and_store_specific_occupation_skills(request.occupation_ids, request.min_skill_level, request.max_skill_level, sources=request.sources)
        task_status[task_id] = {"status": "completed", "file_path": file_path}
    except Exception as e:
        task_status[task_id] = {"status": "failed", "error": str(e)}

@app.post("/train_and_evaluate")
async def train_and_evaluate_api(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """Trains and evaluates a model using the specified dataset."""
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"status": "in_progress", "file_path": ""}
    background_tasks.add_task(train_and_evaluate_model_with_status, request, task_id)
    return {"task_id": task_id, "message": f"Model training started. Check progress using /task_status/{task_id}"}

def train_and_evaluate_model_with_status(request: TrainModelRequest, task_id: str):
    try:
        file_path = train_and_evaluate_model(request.file_path, request.min_category_samples, request.test_size)
        task_status[task_id] = {"status": "completed", "file_path": file_path}
    except Exception as e:
        task_status[task_id] = {"status": "failed", "error": str(e)}

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Retrieves the status of a task by task_id."""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return task_status[task_id]

@app.post("/analyze_occupation")
async def analyze_occupation(data: OccupationRequest, top_n_features: int = Query(10, ge=1, le=100)):
    """Analyzes feature importance for a given occupation."""
    try:
        # Validate URL
        parsed_url = urlparse(data.occupation_id_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        class_label = data.occupation_id_url.split('/')[-1]
        
        # Load feature importance file
        feature_importance_file = output_folder_features / f"feature_importance_{class_label}.csv"
        if not feature_importance_file.exists():
            raise HTTPException(status_code=404, detail=f"Feature importance file not found for occupation {class_label}")
        
        feature_importance = pd.read_csv(feature_importance_file)
        top_features = feature_importance.head(top_n_features).to_dict(orient="records")
        
        # Construct SHAP plot URLs
        bar_plot_name = f"shap_bar_plot_{class_label}.png"
        dot_plot_name = f"shap_dot_plot_{class_label}.png"
        
        result = {
            "message": f"Analysis completed for {class_label}",
            "top_features": top_features,
            "bar_plot": f"/plots/{bar_plot_name}",
            "dot_plot": f"/plots/{dot_plot_name}"
        }
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")

@app.get("/plots/{plot_name}")
async def get_plot(plot_name: str):
    """Retrieves SHAP plot images."""
    if not re.match(r"^[\w\-\.]+$", plot_name):
        raise HTTPException(status_code=400, detail="Invalid plot name")
    
    file_path = plots_folder / plot_name  # The plot name corresponds to the file name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(file_path)

# Run the app using uvicorn (e.g., `uvicorn main:app --reload`)
