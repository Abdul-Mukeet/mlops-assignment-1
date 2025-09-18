# mlops-assignment-1

## Problem Statement
The goal of this assignment is to apply **MLOps practices** to the complete machine learning lifecycle.
We focus on building, training, and tracking multiple machine learning models on a selected dataset, while ensuring reproducibility and monitoring using **MLflow**.
The main objectives are:
* Organize the project with a standard folder structure.
* Train multiple ML models and compare their performance.
* Track experiments, metrics, and artifacts using MLflow.
* Register the best model in MLflow Model Registry.
* Document the entire workflow for reproducibility.
---

## Project Structure
The repository is organized as follows:
mlops-assignment-1/
* data/              # Datasets 
* notebooks/         # Jupyter notebooks for exploration
* src/               # Source code (training, logging, etc.)
  train_models.py
* models/            # Saved trained models
* results/           # Evaluation results, plots, confusion matrices
* screenshots/       # Screenshots of MLflow UI, registry, etc.
* README.md          # Project documentation
* requirements.txt   # Dependencies


## Dataset Description
For this project, we used the **Iris dataset**, a well-known dataset in machine learning.
* **Samples:** 150
* **Features:** 4 (sepal length, sepal width, petal length, petal width)
* **Classes:** 3 types of Iris flowers
  * Iris Setosa
  * Iris Versicolor
  * Iris Virginica
The dataset is included with scikit-learn (`sklearn.datasets.load_iris`) and does not require manual downloading.
---



## Model Selection & Comparison

We trained three different machine learning models on the **Iris dataset (with added Gaussian noise)** to make classification more challenging. The models were compared using **Accuracy, Precision, Recall, and F1-score**.

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.78     | 0.77      | 0.76   | 0.76     |
| Random Forest       | 0.85     | 0.84      | 0.83   | 0.83     |
| SVM                 | 0.91     | 0.91      | 0.91   | 0.90     |

**Best Model:** SVM achieved the highest accuracy and balanced performance across all metrics.
This model has been **registered in the MLflow Model Registry** as `IrisBestModel`.



   ## MLflow Tracking Screenshots

- **Experiments Logged in MLflow**  
  ![MLflow Runs](screenshots/mlflow_runs.png)

- **Confusion Matrix Example**  
  ![Confusion Matrix](screenshots/confusion_matrix.png)

- **Model Registry Example**  
  ![Model Registry](screenshots/model_registry.png)



##Instructions to run the code.
1. **Clone the Repository**
git clone https://github.com/Abdul-Mukeet/mlops-assignment-1.git
cd mlops-assignment-1

2.**Set Up Virtual Environment**
python -m venv venv
venv\Scripts\activate  

3. **Install Dependencies**
pip install -r requirements.txt

4. **Train Models**
Run the training script:
python src/train_models.py

5. **Run MLflow Tracking Server**
Start MLflow UI to log and compare experiments:
mlflow ui --host 0.0.0.0 --port 5000
Open your browser at http://127.0.0.1:5000
 to explore logs, metrics, and model comparisons.

6. **Model Registration**
After training, select the best-performing model from MLflow.
Register it in the MLflow Model Registry.



## Model Registration (Steps)
Model Registration Steps in MLflow
1. Train & Log Models
In your script (src/train_models.py), you already:
Train models
Log metrics/params/artifacts with MLflow
Log the model itself
Example :
mlflow.sklearn.log_model(model, artifact_path="model")

2. Pick the Best Model
After evaluating metrics, select the best model:
best_model = max(results, key=lambda x: x["accuracy"])
print(f"Best Model: {best_model['model_name']} (Accuracy={best_model['accuracy']:.3f})")

3. Register Best Model in MLflow Registry
model_uri = f"runs:/{best_model['run_id']}/model"
mlflow.register_model(model_uri, "iris-classifier")
This creates a new entry in the Model Registry with:
Name: iris-classifier
Version: automatically assigned (v1, v2, etc.)

4. Open MLflow UI and Verify
Run MLflow server:
mlflow ui --host 0.0.0.0 --port 5000
Go to http://127.0.0.1:5000
Navigate to Models tab.
then see your registered model (IrisBestModel).





















