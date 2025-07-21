# Healthcare Analytics Simulation

A machine learning project simulating hospital readmission prediction to support patient care prioritization. Built during Coding Nomads training, this demonstrates data processing, model training, and evaluation skills applicable to AI engineering.

## Key Features & Achievements
- Developed an XGBoost model with Python, Pandas, and Scikit-learn to predict readmission risk, achieving 0.701 AUC-ROC on a dataset of 500,000+ patient records.
- Processed data in BigQuery on GCP, applying statistical analysis for scalability and feature importance visualization using Matplotlib/Seaborn.
- Optimized GitHub repo by excluding large files with recreation instructions.

## Tech Stack
- Python, Pandas, NumPy, Scikit-learn, XGBoost
- GCP BigQuery for data engineering
- Matplotlib/Seaborn for visualizations

## Project Structure
- `notebook/`: Jupyter notebooks for data exploration, model training, and evaluation.
- `data/`: Sample datasets or instructions to recreate (e.g., from Kaggle's MIMIC-III or similar public sources).
- `scripts/`: Any Python scripts for pipelines.

## Setup & Run
1. Clone the repo: `git clone https://github.com/KyleSDeveloper/Healthcare_Analytics_Simulation.git`
2. Install dependencies: `pip install -r requirements.txt` (add a requirements.txt file if missing—list libs like pandas, scikit-learn, xgboost).
3. Run the main notebook: Open in Jupyter and execute.

## Results & Metrics
- AUC-ROC: 0.701
- Visualized risk scores and feature importance for actionable insights in care management.

See my resume for more on how this ties to ethical AI in healthcare.
