from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
from elasticsearch import Elasticsearch, ConnectionError
import logging
from datetime import datetime
import os

# Initialize FastAPI
app = FastAPI()

# Load the model
model = joblib.load('model.pkl')

# Get Elasticsearch host from environment variable
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")

# Initialize Elasticsearch client
try:
    es = Elasticsearch(ELASTICSEARCH_HOST)
    if not es.ping():
        raise ConnectionError("Could not connect to Elasticsearch")
except ConnectionError as e:
    logger.error(f"Elasticsearch connection error: {e}")
    es = None  # Disable Elasticsearch logging if connection fails

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for incoming data validation
class PredictionInput(BaseModel):
    features: list

# Define a model for the retraining request
class RetrainRequest(BaseModel):
    n_neighbors: int = 5
    weights: str = "uniform"  # Can be "uniform" or "distance"
    metric: str = "minkowski"  # Common metric for KNN

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Endpoint for prediction
@app.post("/predict/")
def predict(request: PredictionInput):
    try:
        # Reshape input data for prediction
        input_data = np.array(request.features).reshape(1, -1)
        logger.info(f"Input data shape: {input_data.shape}")

        # Make prediction
        prediction = model.predict(input_data)
        prediction_result = int(prediction[0])
        logger.info(f"Prediction: {prediction_result}")

        # Log prediction to Elasticsearch
        if es:
            log_entry = {
                "input_data": request.features,
                "prediction": prediction_result,
                "timestamp": datetime.utcnow().isoformat()  # Use current timestamp
            }
            try:
                es.index(index="mlflow-predictions", body=log_entry)
                logger.info("Log sent to Elasticsearch")
            except Exception as e:
                logger.error(f"Failed to send log to Elasticsearch: {e}")
        else:
            logger.warning("Elasticsearch is not available. Skipping log.")

        return {"prediction": prediction_result}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Retrain model endpoint
@app.post("/retrain/")
def retrain(request: RetrainRequest):
    try:
        # Load the dataset (replace with your actual data)
        # Example: Replace this with your actual dataset loading logic
        X = np.random.rand(100, 10)  # Example feature data
        y = np.random.randint(0, 2, 100)  # Example target data

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize MLflow experiment
        mlflow.set_experiment("KNN Retraining")
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "n_neighbors": request.n_neighbors,
                "weights": request.weights,
                "metric": request.metric
            })

            # Initialize the KNN model with the new hyperparameters
            knn_model = KNeighborsClassifier(
                n_neighbors=request.n_neighbors,
                weights=request.weights,
                metric=request.metric
            )

            # Train the model
            knn_model.fit(X_train, y_train)

            # Evaluate the model
            predictions = knn_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logger.info(f"Retrained model accuracy: {accuracy}")

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)

            # Save the retrained model
            model_path = "retrained_model.pkl"
            joblib.dump(knn_model, model_path)
            mlflow.log_artifact(model_path)

            # Log retraining event to Elasticsearch
            if es:
                log_entry = {
                    "n_neighbors": request.n_neighbors,
                    "weights": request.weights,
                    "metric": request.metric,
                    "accuracy": accuracy,
                    "timestamp": datetime.utcnow().isoformat()  # Use current timestamp
                }
                try:
                    es.index(index="mlflow-retraining", body=log_entry)
                    logger.info("Retraining log sent to Elasticsearch")
                except Exception as e:
                    logger.error(f"Failed to send retraining log to Elasticsearch: {e}")
            else:
                logger.warning("Elasticsearch is not available. Skipping log.")

        return {"message": "Model retrained successfully", "accuracy": accuracy}

    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Error during retraining: {e}")

# To run the app, use this command in the terminal:
# uvicorn app:app --reload
