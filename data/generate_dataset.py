#!/usr/bin/env python3
"""
Generate synthetic dataset for UGC-AICTE Institutional Performance Tracker
Creates realistic data for 200 institutions from 2019-2023
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_college_names(n=40):
    """Generate realistic Indian college names"""
    prefixes = [
        "Indian Institute of Technology", "National Institute of Technology",
        "Birla Institute of Technology", "Vellore Institute of Technology",
        "Manipal Institute of Technology", "SRM Institute of Science and Technology",
        "Amrita Vishwa Vidyapeetham", "KIIT University", "Lovely Professional University",
        "Kalinga Institute of Industrial Technology", "PSG College of Technology",
        "Thiagarajar College of Engineering", "Anna University", "Jadavpur University",
        "Delhi Technological University", "Netaji Subhas University of Technology",
        "Guru Gobind Singh Indraprastha University", "Jamia Millia Islamia",
        "Aligarh Muslim University", "Banaras Hindu University", "Jawaharlal Nehru University",
        "University of Delhi", "Pune University", "Mumbai University", "Chennai University",
        "Bangalore University", "Hyderabad University", "Kolkata University",
        "Ahmedabad University", "Jaipur University", "Lucknow University",
        "Chandigarh University", "Thapar Institute of Engineering and Technology",
        "Shiv Nadar University", "Ashoka University", "O.P. Jindal Global University",
        "Bennett University", "Plaksha University", "IIIT Hyderabad", "IIIT Bangalore"
    ]
    
    suffixes = [
        "", " - Main Campus", " - North Campus", " - South Campus", 
        " - Technology Campus", " - Engineering College", " - Institute of Technology"
    ]
    
    colleges = []
    for i in range(n):
        if i < len(prefixes):
            name = prefixes[i] + random.choice(suffixes)
        else:
            # Generate additional names for remaining colleges
            regions = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata", "Ahmedabad"]
            types = ["Institute of Technology", "College of Engineering", "University", "Institute of Science"]
            name = f"{random.choice(regions)} {random.choice(types)}"
        colleges.append(name)
    
    return colleges

def generate_synthetic_data():
    """Generate synthetic dataset for institutional performance analysis"""
    
    # Generate college names
    college_names = generate_college_names(40)  # 40 unique colleges
    years = [2019, 2020, 2021, 2022, 2023]
    
    data = []
    
    for college in college_names:
        # Create a base performance profile for each college
        base_performance = np.random.normal(70, 15)  # Mean 70, std 15
        base_performance = max(30, min(95, base_performance))  # Clamp between 30-95
        
        # Determine college tier (affects all metrics)
        if base_performance > 85:
            tier = "tier1"  # Top tier
        elif base_performance > 65:
            tier = "tier2"  # Mid tier
        else:
            tier = "tier3"  # Lower tier
        
        for year in years:
            # Add year-based trends (slight improvement over years)
            year_factor = (year - 2019) * 0.5  # Small improvement each year
            
            # COVID impact in 2020-2021
            covid_impact = -3 if year in [2020, 2021] else 0
            
            # Generate correlated metrics based on tier and base performance
            if tier == "tier1":
                aicte_score = max(75, min(100, base_performance + np.random.normal(5, 8) + year_factor + covid_impact))
                ugc_rating = max(7, min(10, 8 + np.random.normal(0, 1) + year_factor/5))
                nirf_rank = max(1, min(50, int(np.random.normal(25, 15))))
                placement_pct = max(70, min(100, 85 + np.random.normal(0, 10) + year_factor + covid_impact))
                faculty_ratio = round(np.random.uniform(8, 12), 1)
                research_projects = max(15, min(50, int(np.random.normal(30, 10))))
                infrastructure = max(80, min(100, 90 + np.random.normal(0, 8) + year_factor/2))
                satisfaction = max(75, min(100, 85 + np.random.normal(0, 8) + year_factor/2))
                
            elif tier == "tier2":
                aicte_score = max(60, min(90, base_performance + np.random.normal(0, 10) + year_factor + covid_impact))
                ugc_rating = max(5, min(8, 6.5 + np.random.normal(0, 1) + year_factor/5))
                nirf_rank = max(30, min(150, int(np.random.normal(90, 30))))
                placement_pct = max(50, min(90, 70 + np.random.normal(0, 12) + year_factor + covid_impact))
                faculty_ratio = round(np.random.uniform(12, 18), 1)
                research_projects = max(5, min(30, int(np.random.normal(15, 8))))
                infrastructure = max(60, min(90, 75 + np.random.normal(0, 10) + year_factor/2))
                satisfaction = max(60, min(85, 70 + np.random.normal(0, 10) + year_factor/2))
                
            else:  # tier3
                aicte_score = max(40, min(75, base_performance + np.random.normal(-5, 12) + year_factor + covid_impact))
                ugc_rating = max(3, min(6, 4.5 + np.random.normal(0, 1) + year_factor/5))
                nirf_rank = max(100, min(200, int(np.random.normal(160, 25))))
                placement_pct = max(25, min(70, 50 + np.random.normal(0, 15) + year_factor + covid_impact))
                faculty_ratio = round(np.random.uniform(15, 25), 1)
                research_projects = max(0, min(15, int(np.random.normal(5, 5))))
                infrastructure = max(40, min(75, 60 + np.random.normal(0, 12) + year_factor/2))
                satisfaction = max(40, min(75, 55 + np.random.normal(0, 12) + year_factor/2))
            
            # Calculate Final Performance Index (weighted combination)
            performance_index = (
                0.20 * aicte_score +
                0.15 * (ugc_rating * 10) +  # Convert to 0-100 scale
                0.15 * (100 - (nirf_rank - 1) * 100 / 199) +  # Inverse rank score
                0.20 * placement_pct +
                0.05 * (100 / faculty_ratio * 5) +  # Lower ratio is better
                0.10 * (research_projects * 2) +
                0.10 * infrastructure +
                0.05 * satisfaction
            )
            
            # Ensure reasonable bounds
            performance_index = max(30, min(100, performance_index))
            
            data.append({
                'College_Name': college,
                'Year': year,
                'AICTE_Approval_Score': round(aicte_score, 2),
                'UGC_Rating': round(ugc_rating, 2),
                'NIRF_Rank': int(nirf_rank),
                'Placement_Percentage': round(placement_pct, 2),
                'Faculty_Student_Ratio': faculty_ratio,
                'Research_Projects': int(research_projects),
                'Infrastructure_Score': round(infrastructure, 2),
                'Student_Satisfaction_Score': round(satisfaction, 2),
                'Final_Performance_Index': round(performance_index, 2)
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating synthetic UGC-AICTE dataset...")
    
    # Generate dataset
    df = generate_synthetic_data()
    
    # Save to CSV
    output_path = "ugc_aicte_synthetic_dataset_2019_2023.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_path}")
    print("\nDataset Preview:")
    print(df.head())
    print("\nDataset Statistics:")
    print(df.describe())
    print(f"\nUnique colleges: {df['College_Name'].nunique()}")
    print(f"Years covered: {sorted(df['Year'].unique())}")