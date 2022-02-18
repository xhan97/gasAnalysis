# Copyright 2022 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from urllib.parse import urlparse

import mlflow
import numpy as np
from fastapi import BackgroundTasks, FastAPI
from mlflow.tracking import MlflowClient
from src.models.kmeans.data import load_data
#import src.data.make_dataset as pp
#import src.features.build_features as bf
from src.models.kmeans.model import TsKmeans
from src.models.kmeans.train import Trainer

from models import DeleteApiData, PredictApiData, TrainApiData

# mlflow.set_tracking_uri('sqlite:///backend.db')
mlflow.set_tracking_uri("sqlite:///db/backend.db")
app = FastAPI()
mlflowclient = MlflowClient(
    mlflow.get_tracking_uri(), mlflow.get_registry_uri())


def train_model_task(model_name: str, data_path: str, hyperparams: dict):
    """Tasks that trains the model. This is supposed to be running in the background
    Since it's a heavy computation it's better to use a stronger task runner like Celery
    For the simplicity I kept it as a fastapi background task"""

    # Setup env
    # Set MLflow tracking
    mlflow.set_experiment("gasAnalyse")
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Prepare for training
        print("Loading data...")
        train_data = load_data(data_path)

        # Train
        print("Training model")
        model = TsKmeans(hyperparams)
        trainer = Trainer(model)
        model = trainer.train(train_data)

        # Register model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(f"{tracking_url_type_store=}")

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, "dtw_kmeans", registered_model_name=model_name, conda_env=mlflow.sklearn.get_default_conda_env())
        else:
            mlflow.sklearn.log_model(
                model, "dtw_kmeans_m", registered_model_name=model_name)
        # Transition to production. We search for the last model with the name and we stage it to production
        mv = mlflowclient.search_model_versions(
            f"name='{model_name}'")[-1]  # Take last model version
        mlflowclient.transition_model_version_stage(
            name=mv.name, version=mv.version, stage="production")


@app.get("/")
async def read_root():
    return {"Tracking URI": mlflow.get_tracking_uri(),
            "Registry URI": mlflow.get_registry_uri()}


@app.get("/models")
async def get_models_api():
    """Gets a list with model names"""
    model_list = mlflowclient.list_registered_models()
    model_list = [model.name for model in model_list]
    return model_list


@app.post("/train")
async def train_api(data: TrainApiData, background_tasks: BackgroundTasks):
    """Creates a model based on hyperparameters and trains it."""
    hyperparams = data.hyperparams
    model_name = data.model_name
    data_path = data.data_path

    background_tasks.add_task(
        train_model_task, model_name, data_path, hyperparams)

    return {"result": "Training task started"}


@app.post("/predict")
async def predict_api(data: PredictApiData):
    """Predicts on the provided image"""
    img = data.input_image
    model_name = data.model_name
    # Fetch the last model in production
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/Production"
    )
    # Preprocess the image
    # Flatten input, create a batch of one and normalize
    img = np.array(img, dtype=np.float32).flatten()[np.newaxis, ...] / 255
    # Postprocess result
    pred = model.predict(img)
    print(pred)
    res = int(np.argmax(pred[0]))
    return {"result": res}


@app.post("/delete")
async def delete_model_api(data: DeleteApiData):
    model_name = data.model_name
    version = data.model_version

    if version is None:
        # Delete all versions
        mlflowclient.delete_registered_model(name=model_name)
        response = {"result": f"Deleted all versions of model {model_name}"}
    elif isinstance(version, list):
        for v in version:
            mlflowclient.delete_model_version(name=model_name, version=v)
        response = {
            "result": f"Deleted versions {version} of model {model_name}"}
    else:
        mlflowclient.delete_model_version(name=model_name, version=version)
        response = {
            "result": f"Deleted version {version} of model {model_name}"}
    return response
