from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import os

#Initialize app
app = FastAPI()

#Load vectorizer and model
tfidf_vectorizer = joblib.load("model_to_keep/tf_idf_LR/tfidf_vectorizer.pkl")
logistic_regression_model = joblib.load("model_to_keep/tf_idf_LR/tf_idf_LR_model.pkl")

#Request schemas
class TextRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    feedback: str 

#Feedback storage file
feedback_file = "feedback.json"
#Initialize feedback storage if it doesn't exist
if not os.path.exists(feedback_file):
    with open(feedback_file, "w") as f:
        json.dump([], f)

#Temporary storage for the last predicted text
last_predicted_text = {"text": None, "sentiment": None}
last_predicted_sentiment = {"sentiment": None}

#POST route
@app.post("/predict")
async def predict_sentiment(request: TextRequest):
    #Vectorizer
    text_tfidf = tfidf_vectorizer.transform([request.text])
    
    #Prediction + confidence
    prediction = logistic_regression_model.predict(text_tfidf)
    prediction_prob = logistic_regression_model.predict_proba(text_tfidf)[0]
    
    sentiment = "Positive" if prediction == 'positive' else "Negative"
    confidence = max(prediction_prob)
    
    last_predicted_text["text"] = request.text
    last_predicted_text["sentiment"] = sentiment

    return {"text": request.text, "sentiment": sentiment, "confidence": confidence}

@app.post("/feedback")
async def predict_feedback(feedback: FeedbackRequest):
    #Check if there is a text
    if not last_predicted_text["text"]:
        raise HTTPException(status_code=400, detail="No prediction has been made to provide feedback.")

    # Check if the feedback is valid
    if feedback.feedback.lower() != "unsatisfactory prediction":
        raise HTTPException(status_code=400, detail="Invalid feedback type. Only 'unsatisfactory prediction' is allowed.")

    # Save feedback to the feedback.json file
    with open(feedback_file, "r+") as f:
        feedback_data = json.load(f)
        feedback_data.append({"text": last_predicted_text["text"], "sentiment": last_predicted_text["sentiment"],"feedback": feedback.feedback})
        f.seek(0)
        json.dump(feedback_data, f, indent=4)

    # Clear the last predicted text after feedback
    last_predicted_text["text"] = None
    last_predicted_text["sentiment"] = None
    message = {"message": "Feedback submitted successfully", "text": feedback_data[-1]}
    #Clear cache
    feedback_data = None

    return message