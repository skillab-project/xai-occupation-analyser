# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:01:52 2024

@author: Dimitris Tsoukalas (tsoukj)

This function performs the following tasks:

- Occupation IDs Retrieval:
    * Calls the http://skillab-tracker.csd.auth.gr/api/occupations API with 
    additional filters (min_level=3, max_level=3 ) to retrieve all available 
    occupation IDs.
- Job Data Retrieval:    
    * Calls the http://skillab-tracker.csd.auth.gr/api/jobs API for a list of 
    all available occupation IDs.
    * Fetches job-related data, including job ID, upload date, associated skills,
    and the occupation ID.
- Skill Label Retrieval:
    * Extracts unique skill IDs from the job data.
    * Calls the http://skillab-tracker.csd.auth.gr/api/skills API for each unique
    skill ID with additional filters (min_skill_level=4, max_skill_level=4) to 
    retrieve detailed skill information.
- Data Aggregation:
    * Constructs a structured dataset where each row corresponds to a job, and
    each column indicates the presence (1) or absence (0) of a specific skill.
- Output:
    * Stores the aggregated data in a pandas DataFrame, including columns for 
    job ID, upload date, occupation ID, and skill labels.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API = "http://skillab-tracker.csd.auth.gr/api"
USERNAME = os.getenv("SKILLAB_API_USERNAME")
PASSWORD = os.getenv("SKILLAB_API_PASSWORD")

# Define API endpoints
JOBS_URL = f"{API}/jobs"
SKILLS_URL = f"{API}/skills"
OCCUPATIONS_URL = f"{API}/occupations"

# Define the datasets folder path
datasets_folder = Path("datasets")
datasets_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist

def get_token() -> str:
    res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD})
    res.raise_for_status()
    return res.text.strip().replace('"', "")

def fetch_and_store_all_occupation_skills(min_level: int = 3, max_level: int = 3, min_skill_level: int = 4, max_skill_level: int = 4):
    """
    Fetches occupation, job, and skill data from the Skillab Tracker API and stores it as a CSV file.

    Parameters:
        min_level (int): Minimum level filter for occupations.
        max_level (int): Maximum level filter for occupations.
        min_skill_level (int): Minimum skill level filter for skills.
        max_skill_level (int): Maximum skill level filter for skills.

    Returns:
        str: Path to the saved CSV file.
    """
    try:
        token = get_token()
        HEADERS = {"Authorization": f"Bearer {token}"}
        payload = {"min_level": min_level, "max_level": max_level}
        
        # Initialize containers for aggregated occupation data
        occupation_ids_list = []
        page = 1
        
        while True:
            # Make the API call with pagination
            response = requests.post(f"{OCCUPATIONS_URL}?page={page}", data=payload, headers=HEADERS)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Parse JSON response
            response_data = response.json()
            
            # Extract occupation IDs
            new_occupation_ids = [item["id"] for item in response_data.get("items", [])]
            occupation_ids_list.extend(new_occupation_ids)
            
            # Check if there are more pages
            total_count = response_data.get("count", 0)
            if len(occupation_ids_list) >= total_count:
                break
            
            page += 1
        
        # Initialize containers for aggregated job and skill data
        all_jobs_data = []
        unique_skills = set()
        
        for occupation_id in occupation_ids_list:
            payload = {"occupation_ids": occupation_id}
            page = 1
            
            while True:
                # Call the jobs API with pagination
                response = requests.post(f"{JOBS_URL}?page={page}", data=payload, headers=HEADERS)
                response.raise_for_status()  # Check if the request was successful
                
                # Parse the JSON response
                data = response.json()
                
                # Extract job items
                job_items = data.get("items", [])
                for item in job_items:
                    all_jobs_data.append({
                        "job_id": item.get("id"),
                        "upload_date": item.get("upload_date"),
                        "occupation4d": occupation_id,
                        "skills": item.get("skills", [])
                    })
                    unique_skills.update(item.get("skills", []))
                
                # Check if there are more pages
                total_count = data.get("count", 0)
                if len(all_jobs_data) >= total_count:
                    break
                
                page += 1
        
        # Fetch skill details for each unique skill
        skill_labels = {}
        for skill_id in unique_skills:
            skill_payload = {"ids": skill_id, "min_skill_level": min_skill_level, "max_skill_level": max_skill_level}
            page = 1
            
            while True:
                # Call the skills API with pagination
                skill_response = requests.post(f"{SKILLS_URL}?page={page}", data=skill_payload, headers=HEADERS)
                skill_response.raise_for_status()
                skill_data = skill_response.json()
                
                # Extract the label of the skill
                skill_items = skill_data.get("items", [])
                for skill_item in skill_items:
                    skill_labels[skill_item["id"]] = skill_item.get("label", "Unknown")
                
                # Check if there are more pages
                total_count = skill_data.get("count", 0)
                if len(skill_labels) >= total_count:
                    break
    
                page += 1
        
        # Prepare the dataframe
        df_rows = []
        for job in all_jobs_data:
            row = {
                "job_id": job["job_id"],
                "upload_date": job["upload_date"],
                "occupation4d": job["occupation4d"]
            }
            for skill_id in job["skills"]:
                skill_label = skill_labels.get(skill_id)
                if skill_label:
                    row[skill_label] = 1  # Only storing 1s, assuming missing = 0
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        df.fillna(0, inplace=True)
        for col in df.columns:
            if col not in ["job_id", "upload_date", "occupation4d"]:
                df[col] = df[col].astype(pd.SparseDtype("int", fill_value=0))  # Apply SparseDtype only to skills
        
        # Save the dataframe to csv
        csv_file_path = datasets_folder / "all_occupation4d_skills.csv"
        df.to_csv(csv_file_path, index=False)
        
        print(f"Dataset saved successfully at {csv_file_path}")
        return str(csv_file_path)
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while calling the API: {e}")
        return None
