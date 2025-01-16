# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:01:52 2024

@author: Dimitris Tsoukalas (tsoukj)

This script performs the following tasks:

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

import requests
import pandas as pd

# Define the API endpoint for jobs
jobs_url = "http://skillab-tracker.csd.auth.gr/api/jobs"
# Define the API endpoint for skills
skills_url = "http://skillab-tracker.csd.auth.gr/api/skills"

# List of occupation IDs to include in the request body
occupation_ids_list = [
    "http://data.europa.eu/esco/isco/C2511",
    "http://data.europa.eu/esco/isco/C2512",
    "http://data.europa.eu/esco/isco/C2513",
    "http://data.europa.eu/esco/isco/C2514",
    "http://data.europa.eu/esco/isco/C2519"
]

try:
    # Initialize containers for aggregated job and skill data
    all_jobs_data = []
    unique_skills = set()
    
    for occupation_id in occupation_ids_list:
        payload = {"occupation_ids": occupation_id}
        page = 1
        
        while True:
            # Call the jobs API with pagination
            response = requests.post(f"{jobs_url}?page={page}", data=payload)
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
        skill_payload = {
            "ids": skill_id,
            "min_skill_level": 4,
            "max_skill_level": 4
        }
        page = 1
        
        while True:
            # Call the skills API with pagination
            skill_response = requests.post(f"{skills_url}?page={page}", data=skill_payload)
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
    all_skill_labels = list(skill_labels.values())
    df_data = []

    for job in all_jobs_data:
        row = {
            "job_id": job["job_id"],
            "upload_date": job["upload_date"],
            "occupation4d": job["occupation4d"]
        }
        for skill_label in all_skill_labels:
            row[skill_label] = 0

        for skill_id in job["skills"]:
            skill_label = skill_labels.get(skill_id)
            if skill_label:
                row[skill_label] = 1

        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Print or save the dataframe
    # print(df)
    df.to_csv("software and applications developers and analysts_occupation4d_skills.csv", index=False)

except requests.exceptions.RequestException as e:
    print(f"Error occurred while calling the API: {e}")
