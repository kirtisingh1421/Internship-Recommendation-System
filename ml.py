import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# Step 1: Load Internship Dataset
# -------------------------------
df = pd.read_csv("internship.csv")

# -------------------------------
# Step 2: Preprocess Internship Data
# -------------------------------
# Convert REQUIRED_SKILLS column to array
df["Skills_array"] = df["REQUIRED_SKILLS"].fillna("").apply(lambda x: [s.strip() for s in x.split(",")])

# Convert DURATION like "3 Months" -> 3
df["Duration_num"] = df["DURATION"].str.extract(r"(\d+)").fillna(0).astype(float)

# Map EXPERIENCE_LEVEL strings to numbers
experience_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
df["EXPERIENCE_LEVEL_ENCODED"] = df["EXPERIENCE_LEVEL"].map(experience_map).fillna(0)

# Encode EDUCATION_LEVEL to numeric using a global encoder
edu_encoder_global = LabelEncoder()
df["Education_encoded"] = edu_encoder_global.fit_transform(df["EDUCATION_LEVEL"].astype(str))

# -------------------------------
# Step 3: Encode Skills as Binary Vectors
# -------------------------------
all_skills = sorted({skill.strip() for skills in df["Skills_array"] for skill in skills if skill})

def encode_skills(skill_list):
    # Ensure skills are stripped of whitespace before checking against all_skills
    stripped_skills = {s.strip() for s in skill_list}
    return [1 if skill in stripped_skills else 0 for skill in all_skills]

df["Skills_vector"] = df["Skills_array"].apply(encode_skills)

# -------------------------------
# Step 4: Normalize Numeric Features
# -------------------------------
scaler = MinMaxScaler()
df[["Duration_scaled", "Experience_scaled", "Education_scaled"]] = scaler.fit_transform(
    df[["Duration_num", "EXPERIENCE_LEVEL_ENCODED", "Education_encoded"]]
)

# -------------------------------
# Step 5: Build Weighted Feature Matrix
# -------------------------------
# Weighting: skills=5x, duration=3x, experience=1x, education=1x
feature_matrix = np.array([
    list(np.array(sv)*5) + [dur*3, exp*1, edu*1] 
    for sv, dur, exp, edu in zip(df["Skills_vector"], df["Duration_scaled"], df["Experience_scaled"], df["Education_scaled"])
])

# -------------------------------
# Step 6: Train NearestNeighbors Model
# -------------------------------
knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_model.fit(feature_matrix)

# -------------------------------
# Step 7: Recommendation Function
# -------------------------------
def recommend_internships(student_skills, student_duration, student_experience, student_education, top_n=5):
    # Encode student education using the global encoder
    try:
        student_edu_encoded = edu_encoder_global.transform([student_education])[0]
    except ValueError:
        # Handle unseen education levels
        student_edu_encoded = -1  # A value to represent "unseen"

    # Normalize student numeric features
    student_duration_scaled, student_experience_scaled, student_edu_scaled = scaler.transform(
        [[student_duration, student_experience, student_edu_encoded]]
    )[0]

    # Encode skills and apply weight
    student_vector = list(np.array(encode_skills(student_skills))*5) + [student_duration_scaled*3, student_experience_scaled*1, student_edu_scaled*1]
    student_vector = np.array(student_vector).reshape(1, -1)

    # Find nearest internships
    distances, indices = knn_model.kneighbors(student_vector, n_neighbors=top_n)
    
    recommended = df.iloc[indices[0]].copy()
    
    # Calculate match score
    max_dist = distances.max() if distances.max() > 0 else 1
    recommended["Match_Score"] = 100 - (distances[0] / max_dist * 100)
    
    return recommended[[
        "TITLE", "COMPANY_NAME", "REQUIRED_SKILLS", "Duration_num",
        "EXPERIENCE_LEVEL", "EDUCATION_LEVEL", "Match_Score"
    ]]

# -------------------------------
# Step 8: Test Example
# -------------------------------
if __name__ == "_main_":
    student_skills = ["Python", "SQL"]
    student_duration = 4
    student_experience = experience_map["Intermediate"] # Use the mapped value
    student_education = "Bachelor"

    top_internships = recommend_internships(
        student_skills, student_duration, student_experience, student_education, top_n=5
    )
    print(top_internships)