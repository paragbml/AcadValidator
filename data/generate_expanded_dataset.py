#!/usr/bin/env python3
"""
Generate expanded synthetic dataset with 150+ top Indian colleges
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Top 150 Indian Engineering/Technical Colleges
indian_colleges = [
    # IITs
    "IIT Delhi", "IIT Bombay", "IIT Kanpur", "IIT Kharagpur", "IIT Chennai", 
    "IIT Roorkee", "IIT Guwahati", "IIT Hyderabad", "IIT Indore", "IIT Bhubaneswar",
    "IIT Gandhinagar", "IIT Jodhpur", "IIT Patna", "IIT Ropar", "IIT Mandi",
    "IIT (ISM) Dhanbad", "IIT Palakkad", "IIT Tirupati", "IIT Bhilai", "IIT Goa",
    "IIT Jammu", "IIT Dharwad",
    
    # NITs
    "NIT Tiruchirappalli", "NIT Warangal", "NIT Surathkal", "NIT Calicut", 
    "NIT Rourkela", "NIT Allahabad", "NIT Bhopal", "NIT Nagpur", "NIT Kurukshetra",
    "NIT Durgapur", "NIT Jamshedpur", "NIT Silchar", "NIT Hamirpur", "NIT Jalandhar",
    "NIT Raipur", "NIT Agartala", "NIT Patna", "NIT Meghalaya", "NIT Manipur",
    "NIT Mizoram", "NIT Sikkim", "NIT Arunachal Pradesh", "NIT Uttarakhand",
    "NIT Delhi", "NIT Goa", "NIT Puducherry", "NIT Andhra Pradesh", "NIT Karnataka",
    
    # IIITs
    "IIIT Hyderabad", "IIIT Allahabad", "IIIT Bangalore", "IIIT Gwalior", 
    "IIIT Jabalpur", "IIIT Kota", "IIIT Sri City", "IIIT Vadodara", "IIIT Nagpur",
    "IIIT Pune", "IIIT Kurnool", "IIIT Una", "IIIT Sonepat", "IIIT Kalyani",
    "IIIT Tiruchirappalli", "IIIT Dharwad", "IIIT Lucknow", "IIIT Ranchi",
    "IIIT Kottayam", "IIIT Manipur", "IIIT Agartala", "IIIT Raichur",
    
    # Private Universities
    "Vellore Institute of Technology", "Birla Institute of Technology", 
    "SRM Institute of Science and Technology", "Amity University", 
    "Manipal Institute of Technology", "Thapar Institute of Engineering",
    "Lovely Professional University", "Kalinga Institute of Industrial Technology",
    "Bharati Vidyapeeth University", "Vel Tech Rangarajan Dr Sagunthala R&D Institute",
    "Hindustan Institute of Technology", "SSN College of Engineering",
    "PSG College of Technology", "Thiagarajar College of Engineering",
    "Kumaraguru College of Technology", "Bannari Amman Institute of Technology",
    "Sri Krishna College of Engineering", "Sastra University", "Anna University",
    "Shanmugha Arts Science Technology",
    
    # State Universities & Engineering Colleges
    "Delhi Technological University", "Jadavpur University", "Bengal Engineering College",
    "College of Engineering Pune", "Government College of Technology Coimbatore",
    "Visvesvaraya National Institute of Technology", "Indian Institute of Engineering Science",
    "Motilal Nehru National Institute of Technology", "Maulana Azad National Institute of Technology",
    "Sardar Vallabhbhai National Institute of Technology", "Dr. B R Ambedkar National Institute of Technology",
    "Malaviya National Institute of Technology", "Pandit Dwarka Prasad Mishra Indian Institute",
    "Rajiv Gandhi Institute of Technology", "Government Engineering College",
    "Netaji Subhas Institute of Technology", "Indira Gandhi Delhi Technical University",
    "Guru Gobind Singh Indraprastha University", "Jamia Millia Islamia University",
    "Aligarh Muslim University", "Banaras Hindu University", "Osmania University",
    "Andhra University", "Jawaharlal Nehru Technological University", "Kakatiya University",
    "Sri Venkateswara University", "Acharya Nagarjuna University", "Krishna University",
    "Rajiv Gandhi University of Knowledge Technologies", "Indian Institute of Information Technology",
    "Birla Institute of Technology and Science", "International Institute of Information Technology",
    "Dhirubhai Ambani Institute of Information Technology", "Institute of Chemical Technology",
    "Harcourt Butler Technical University", "Uttar Pradesh Technical University",
    "Dr. A.P.J. Abdul Kalam Technical University", "Rajasthan Technical University",
    "Gujarat Technological University", "Anna University of Technology",
    "Visvesvaraya Technological University", "Jawaharlal Nehru Technological University Hyderabad",
    "Jawaharlal Nehru Technological University Kakinada", "Jawaharlal Nehru Technological University Anantapur",
    "Kalasalingam Academy of Research and Education", "Karunya Institute of Technology",
    "VIT University", "SRM University", "Amrita Vishwa Vidyapeetham", "Christ University",
    "PES University", "R.V. College of Engineering", "BMS College of Engineering",
    "M.S. Ramaiah Institute of Technology", "Dayananda Sagar College of Engineering",
    "Sir M. Visvesvaraya Institute of Technology", "New Horizon College of Engineering",
    "Nitte Meenakshi Institute of Technology", "B.N.M. Institute of Technology",
    "The Oxford College of Engineering", "Sapthagiri College of Engineering",
    "Siddaganga Institute of Technology", "Malnad College of Engineering",
    "National Institute of Engineering", "Mysore University", "Bangalore University",
    "Kuvempu University", "Mangalore University", "Gulbarga University",
    "Karnatak University", "Belgaum University", "Tumkur University",
    "Davangere University", "Hassan University", "Ballari University",
    "Raichur University", "Bidar University", "Bagalkot University",
    "Vijayapura University", "Uttara Kannada University", "Chamarajanagar University",
    "Mandya University", "Chikmagalur University", "Kodagu University",
    "Udupi University", "Dakshina Kannada University", "Shimoga University"
]

# Ensure we have exactly 150 colleges
if len(indian_colleges) > 150:
    indian_colleges = indian_colleges[:150]
elif len(indian_colleges) < 150:
    # Add some more generic names to reach 150
    for i in range(len(indian_colleges), 150):
        indian_colleges.append(f"Engineering College {i-len(indian_colleges)+1}")

def create_expanded_dataset():
    """Create expanded dataset with 150 colleges across 5 years"""
    years = [2019, 2020, 2021, 2022, 2023]
    data = []
    
    # Define college tiers for realistic performance distribution
    tier_1_colleges = indian_colleges[:25]  # Top 25 - IITs, top NITs
    tier_2_colleges = indian_colleges[25:75]  # Next 50 - NITs, IIITs, top private
    tier_3_colleges = indian_colleges[75:]   # Remaining - state universities, private
    
    for college in indian_colleges:
        # Determine college tier
        if college in tier_1_colleges:
            tier = 1
            base_performance = np.random.uniform(85, 95)
        elif college in tier_2_colleges:
            tier = 2
            base_performance = np.random.uniform(70, 85)
        else:
            tier = 3
            base_performance = np.random.uniform(55, 75)
        
        for year in years:
            # Add year-over-year variation
            year_factor = (year - 2019) * 0.5  # Slight improvement over years
            performance_variation = np.random.uniform(-3, 3)
            final_performance = base_performance + year_factor + performance_variation
            final_performance = max(30, min(100, final_performance))  # Clamp between 30-100
            
            # Generate correlated metrics based on performance
            aicte_score = final_performance + np.random.uniform(-5, 5)
            aicte_score = max(40, min(100, aicte_score))
            
            ugc_rating = (final_performance / 100) * 8 + 2 + np.random.uniform(-0.5, 0.5)
            ugc_rating = max(3, min(10, ugc_rating))
            
            # NIRF rank - lower is better, so inverse relationship
            nirf_rank = int(200 - (final_performance - 30) * 2.4 + np.random.uniform(-20, 20))
            nirf_rank = max(1, min(200, nirf_rank))
            
            placement_pct = final_performance + np.random.uniform(-8, 8)
            placement_pct = max(25, min(100, placement_pct))
            
            # Faculty ratio - better colleges have lower ratios
            faculty_ratio = 25 - (final_performance - 30) * 0.24 + np.random.uniform(-2, 2)
            faculty_ratio = max(8, min(25, faculty_ratio))
            
            research_projects = int((final_performance - 30) * 0.7 + np.random.uniform(-5, 5))
            research_projects = max(0, min(50, research_projects))
            
            infrastructure_score = final_performance + np.random.uniform(-3, 3)
            infrastructure_score = max(40, min(100, infrastructure_score))
            
            satisfaction_score = final_performance + np.random.uniform(-5, 5)
            satisfaction_score = max(40, min(100, satisfaction_score))
            
            record = {
                'College_Name': college,
                'Year': year,
                'AICTE_Approval_Score': round(aicte_score, 2),
                'UGC_Rating': round(ugc_rating, 2),
                'NIRF_Rank': nirf_rank,
                'Placement_Percentage': round(placement_pct, 2),
                'Faculty_Student_Ratio': round(faculty_ratio, 1),
                'Research_Projects': research_projects,
                'Infrastructure_Score': round(infrastructure_score, 2),
                'Student_Satisfaction_Score': round(satisfaction_score, 2),
                'Final_Performance_Index': round(final_performance, 2)
            }
            data.append(record)
    
    return pd.DataFrame(data)

def main():
    """Generate and save the expanded dataset"""
    print("🏫 Generating Expanded Indian Colleges Dataset")
    print(f"📊 Creating data for {len(indian_colleges)} colleges across 5 years")
    print(f"📈 Total records: {len(indian_colleges) * 5}")
    
    # Generate dataset
    df = create_expanded_dataset()
    
    # Save to CSV
    output_file = '../data/expanded_indian_colleges_dataset_2019_2023.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Dataset saved to: {output_file}")
    print(f"📋 Dataset shape: {df.shape}")
    print(f"📈 Performance range: {df['Final_Performance_Index'].min():.2f} - {df['Final_Performance_Index'].max():.2f}")
    print(f"🏆 Unique colleges: {df['College_Name'].nunique()}")
    
    # Display sample data
    print("\n📝 Sample records:")
    print(df.head(10))
    
    # Display college tier breakdown
    tier_1_count = len([c for c in df['College_Name'].unique() if c in indian_colleges[:25]])
    tier_2_count = len([c for c in df['College_Name'].unique() if c in indian_colleges[25:75]])
    tier_3_count = len([c for c in df['College_Name'].unique() if c in indian_colleges[75:]])
    
    print(f"\n🏆 College Distribution:")
    print(f"   Tier 1 (Premium): {tier_1_count} colleges")
    print(f"   Tier 2 (Good): {tier_2_count} colleges") 
    print(f"   Tier 3 (Average): {tier_3_count} colleges")

if __name__ == "__main__":
    main()