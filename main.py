import uvicorn
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from fastapi_utils.tasks import repeat_every

from utils import get_data, verify_secret_key

app = FastAPI()

PORT = int(os.getenv("PORT"))


origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and preprocess your data
df = pd.DataFrame(get_data())
# Combine text fields into a single column
df['combined_text'] = df['responsibilities'] + " " + df['minimum_qualifications'] + " " + df['preferred_qualifications']
# Initialize and configure necessary models and transformers
vectorizer = TfidfVectorizer(max_features=1000)  # Configure the number of max features
nmf_model = NMF(n_components=10)  # Number of components for dimensionality reduction
normalizer = Normalizer()
pipeline = make_pipeline(vectorizer, nmf_model, normalizer)
location_weight = 10  # Adjust this factor as needed


# Train the initial model
def train_model():
    global df, pipeline

    # Process the combined text data
    text_features = pipeline.fit_transform(df['combined_text'])

    # Scale location features more heavily
    location_features = df[['latitude', 'longitude']].values * location_weight

    # Combine the text features with the location features
    combined_features = np.hstack([text_features, location_features])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined_features)

    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(features_scaled)

    return knn, scaler, features_scaled


knn_model, scaler, features_scaled = train_model()


class JobQuery(BaseModel):
    latitude: float
    longitude: float
    description: str


@app.on_event("startup")
@repeat_every(seconds=60 * 60 * 24)  # Repeat every 24 hours
async def startup_event():
    global df, pipeline, knn_model, features_scaled
    # Reload or retrain models if necessary
    knn_model, scaler, features_scaled = train_model()

    print("Models trained successfully")


@app.post("/retrain")
async def retrain(secret_key: str = Depends(verify_secret_key)):
    global knn_model, scaler, features_scaled
    knn_model, scaler, features_scaled = train_model()
    return {"message": "Models retrained successfully"}


@app.get("/getRecommendations")
async def get_recommendations(query: JobQuery, secret_key: str = Depends(verify_secret_key)):
    global df, knn_model, features_scaled
    # Process user description for similarity
    user_description_vector = pipeline.transform([query.description])

    # Calculate geographical distances and normalize
    user_location = np.array([[query.latitude, query.longitude]]) * location_weight
    user_combined_features = np.hstack([user_description_vector, user_location])

    user_features_scaled = scaler.transform(user_combined_features)

    # Get nearest jobs
    distances, indices = knn_model.kneighbors(user_features_scaled)
    nearest_jobs = df.iloc[indices[0]].to_dict(orient='records')

    return {"recommendations": nearest_jobs}

@app.get("/test")
async def test(secret_key: str = Depends(verify_secret_key)):
    return {"message": "Test successful"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

