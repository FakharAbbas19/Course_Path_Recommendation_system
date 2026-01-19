from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("decision_tree_model.pkl")

# Define request body schema
class StudentInput(BaseModel):
    cgpa: float
    course_count: int
    interest: int   # 👈 Ab integer rakho, string nahi

@app.post("/predict")
def predict(data: StudentInput):
    try:
        # Directly use integer interest
        prediction = model.predict([[data.cgpa, data.course_count, data.interest]])
        return {"Recommended Course": str(prediction[0])}
    except Exception as e:
        return {"error": str(e)}