from flask import Flask, request, jsonify
import pymysql
import json
import os
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# ----------------------------------
# Flask App Setup
# ----------------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------------
# MySQL connection 
# ----------------------------------
try:
    db = pymysql.connect(
        host="localhost",
        user="root",
        password="Khushi07@",
        database="intern",
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    print("Connected to MySQL successfully.")
except pymysql.MySQLError as e:
    print(f"Error connecting to database: {e}")
    db = None

# JSON backup file
JSON_FILE = "students.json"
if not os.path.exists(JSON_FILE):
    with open(JSON_FILE, "w") as f:
        json.dump([], f)

# ----------------------------------
# ML Model Loading & Preprocessing
# ----------------------------------
try:
    df = pd.read_csv("internship.csv")
    df.columns = [c.replace(' ', '_').replace('.', '') for c in df.columns]

    df["Skills_array"] = df["REQUIRED_SKILLS"].fillna("").apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])

    eligibility_map = {"Undergraduates": 0, "Graduates": 1, "Postgraduates": 2}
    df["ELIGIBILITY_ENCODED"] = df["ELIGIBILITY"].map(eligibility_map).fillna(0)

    df["Experience_encoded"] = df["ELIGIBILITY_ENCODED"]
    df["Education_encoded"] = df["ELIGIBILITY_ENCODED"]

    # Extract numerical duration
    def extract_duration(duration_str):
        if "Above" in duration_str:
            return 7.0
        parts = [float(s) for s in duration_str.split() if s.replace('.', '', 1).isdigit()]
        return sum(parts) / len(parts) if parts else 0.0

    df["Duration_num"] = df["DURATION"].apply(extract_duration)

    all_skills = sorted({skill for skills in df["Skills_array"] for skill in skills})
    def encode_skills(skill_list):
        return [1 if skill in skill_list else 0 for skill in all_skills]

    df["Skills_vector"] = df["Skills_array"].apply(encode_skills)

    scaler = MinMaxScaler()
    df[["Duration_scaled", "Experience_scaled", "Education_scaled"]] = scaler.fit_transform(
        df[["Duration_num", "Experience_encoded", "Education_encoded"]]
    )

    feature_matrix = np.array([
        list(np.array(sv) * 5) + [dur * 3, exp * 1, edu * 1]
        for sv, dur, exp, edu in zip(df["Skills_vector"], df["Duration_scaled"], df["Experience_scaled"], df["Education_scaled"])
    ])

    knn_model = NearestNeighbors(n_neighbors=10, metric='euclidean')
    knn_model.fit(feature_matrix)

    ml_ready = True
    print("ML model loaded and trained successfully.")
except Exception as e:
    print(f"Error loading ML model or data: {e}")
    ml_ready = False

# ----------------------------------
# Add Student API
# ----------------------------------
@app.route("/add_student", methods=["POST"])
def add_student():
    if db is None:
        return jsonify({"message": "Database connection failed"}), 500

    data = request.json
    required_fields = ["name", "age", "experience_level", "education_level", "skills", "preferred_duration"]
    if not all(field in data and data.get(field) is not None for field in required_fields):
        return jsonify({"message": "Missing one or more required fields"}), 400

    try:
        age = int(data.get("age", 0))
    except (ValueError, TypeError):
        return jsonify({"message": "Age must be a valid number."}), 400

    try:
        skills_list = data.get("skills", [])
        skills_json = json.dumps(skills_list)

        query = """
        INSERT INTO student 
        (NAME, AGE, EXPERIENCE_LEVEL, EDUCATION_LEVEL, SKILLS, PREFERRED_DURATION)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (
            data.get("name"),
            age,
            data.get("experience_level"),
            data.get("education_level"),
            skills_json,
            data.get("preferred_duration")
        )

        cursor = db.cursor()
        cursor.execute(query, values)
        db.commit()
        cursor.close()

        with open(JSON_FILE, "r+") as f:
            try:
                students = json.load(f)
            except json.JSONDecodeError:
                students = []
            students.append(data)
            f.seek(0)
            json.dump(students, f, indent=4)
            f.truncate()

        return jsonify({"message": "✅ Student data saved in MySQL + JSON!"})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": str(e)}), 500

# ----------------------------------
# ML Recommendation API
# ----------------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    if not ml_ready:
        return jsonify({"message": "Recommendation model is not ready."}), 503

    try:
        data = request.json
        user_skills = set(data.get("skills", []))
        user_duration = data.get("preferred_duration", "0 Months")
        user_experience = data.get("experience_level", "Beginner")
        user_education = data.get("education_level", "Any")

        eligibility_map = {"Undergraduates": 0, "Graduates": 1, "Postgraduates": 2}
        user_experience_map = {"Beginner": "Undergraduates", "Intermediate": "Graduates", "Advanced": "Postgraduates"}
        user_education_map = {
            "High School": "Undergraduates",
            "Associate": "Undergraduates",
            "Bachelor": "Graduates",
            "Master": "Postgraduates",
            "PhD": "Postgraduates",
            "Any": "Undergraduates"
        }

        student_experience_encoded = eligibility_map.get(user_experience_map.get(user_experience, "Undergraduates"), 0)
        student_education_encoded = eligibility_map.get(user_education_map.get(user_education, "Undergraduates"), 0)

        def extract_duration_from_input(duration_str):
            if "Above" in duration_str:
                return 7.0
            parts = [float(s) for s in duration_str.split() if s.replace('.', '', 1).isdigit()]
            return sum(parts) / len(parts) if parts else 0.0

        student_duration = extract_duration_from_input(user_duration)

        student_skills_vector = list(np.array(encode_skills(user_skills)) * 5)

        student_vector_scaled = scaler.transform(
            [[student_duration, student_experience_encoded, student_education_encoded]]
        )[0]

        student_vector = np.array(student_skills_vector + [student_vector_scaled[0] * 3, student_vector_scaled[1] * 1, student_vector_scaled[2] * 1]).reshape(1, -1)

        distances, indices = knn_model.kneighbors(student_vector, n_neighbors=10)
        recommended = df.iloc[indices[0]].copy()

        # -------------------------------
        # Duration Filtering
        # -------------------------------
        if "Above" in user_duration:
            recommended = recommended[recommended["DURATION"].str.contains("Above", case=False, na=False)]
        elif "1 - 3" in user_duration:
            recommended = recommended[recommended["DURATION"].str.contains("1", case=False, na=False)]
        elif "3 - 6" in user_duration:
            recommended = recommended[recommended["DURATION"].str.contains("3", case=False, na=False) | recommended["DURATION"].str.contains("6", case=False, na=False)]

        if recommended.empty:  # fallback
            recommended = df.iloc[indices[0]].copy()

        max_dist = distances.max() if distances.max() > 0 else 1
        recommended["Match_Score"] = 100 - (distances[0][:len(recommended)] / max_dist * 100)

        output = recommended[[
            "TITLE", "COMPANY_NAME", "REQUIRED_SKILLS", "DURATION", "ELIGIBILITY", "Match_Score"
        ]].rename(columns={
            "REQUIRED_SKILLS": "skills",
            "DURATION": "duration",
            "ELIGIBILITY": "eligibility"
        }).to_dict(orient="records")

        return jsonify(output)

    except Exception as e:
        print(f"Error in recommend function: {e}")
        return jsonify({"message": "Error processing recommendation request"}), 500

# ----------------------------------
# Run Flask
# ----------------------------------
if __name__ == "_main_":
    app.run(debug=True, port=5000)