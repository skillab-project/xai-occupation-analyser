# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:23:34 2025

@author: Dimitris Tsoukalas (tsoukj)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import re
from urllib.parse import urlparse

app = FastAPI()

# Define folders
output_folder_features = Path("feature_importance")
plots_folder = Path("plots")
output_folder_features.mkdir(exist_ok=True)
plots_folder.mkdir(exist_ok=True)

# Define input schema for the API
class OccupationRequest(BaseModel):
    occupation_id_url: str

@app.post("/analyze_occupation")
async def analyze_occupation(data: OccupationRequest, top_n_features: int = Query(10, ge=1, le=100)):
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
    # Validate plot name
    if not re.match(r"^[\w\-\.]+$", plot_name):
        raise HTTPException(status_code=400, detail="Invalid plot name")
    
    file_path = plots_folder / plot_name  # The plot name corresponds to the file name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(file_path)

# Run the app using uvicorn (e.g., `uvicorn main:app --reload`)
