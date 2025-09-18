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
| SVM                 | 0.89     | 0.88      | 0.87   | 0.88     |

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

1. **Start MLflow Tracking Server**  
   Run the following command in terminal to start MLflow UI:  
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000





















