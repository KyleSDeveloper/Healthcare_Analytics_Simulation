# 🏥 Predicting Hospital Readmissions with Machine Learning

This project predicts patient hospital readmission using the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008). It applies advanced machine learning techniques to identify whether a patient is likely to be readmitted:

- **0** → Not readmitted  
- **1** → Readmitted within 30 days  
- **2** → Readmitted after 30 days  

---

## 🎯 Project Goals

- Clean and preprocess a real-world healthcare dataset
- Perform exploratory data analysis (EDA)
- Engineer relevant features
- Handle class imbalance
- Build a multiclass classification model using **XGBoost**
- Tune hyperparameters using **Optuna**
- Evaluate and interpret model performance

---

## 🧠 Machine Learning Approach

### ✅ Techniques Used

- XGBoost multiclass classification
- Sample weighting to address class imbalance
- Hyperparameter tuning with Optuna
- Performance evaluation with classification reports and confusion matrices

### 🧪 Label Encoding Logic

| Label | Meaning                  |
|-------|--------------------------|
| 0     | Not readmitted           |
| 1     | Readmitted within 30 days |
| 2     | Readmitted after 30 days |

---

## 📊 Results

| Metric         | Value    |
|----------------|----------|
| Accuracy       | 52.0%    |
| Macro F1 Score | 0.45     |
| Class 1 Recall | 31.0%    |
| Model          | Tuned XGBoost (via Optuna) |

- Optuna tuning improved macro F1 and recall for class 1 (early readmission)
- Class imbalance was addressed using sample weighting

---

## 📁 Project Structure

Healthcare_Analytics_Simulation/
├── data/ # Raw and sample data (not tracked in Git)
├── notebooks/
│ ├── 01_EDA.ipynb # Exploratory data analysis
│ ├── 02_Modeling_XGBoost.ipynb # Initial modeling attempts
│ └── 03_Hyperparameter_Tuning_Optuna.ipynb
├── models/
│ └── best_xgb_model.json # Trained model
├── src/
│ ├── preprocessing.py # Feature engineering and encoding
│ ├── train_model.py # Model training script
├── requirements.txt
├── README.md
└── .gitignore


---

## ⚙️ How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Healthcare_Analytics_Simulation.git
cd Healthcare_Analytics_Simulation

2. Install dependencies

pip install -r requirements.txt

3. Run notebooks

Open notebooks in Jupyter or VSCode to explore and reproduce results.
📦 Requirements

    Python 3.12+

    XGBoost

    Scikit-learn

    Optuna

    Pandas, NumPy, Matplotlib

Install all requirements:

pip install -r requirements.txt

📌 Key Learnings

    How to handle class imbalance in multiclass problems

    How to tune hyperparameters using Optuna

    How to balance precision/recall tradeoffs in clinical data

    How to structure and document ML projects for recruiters

📜 License

This project is for educational and portfolio purposes only. Not intended for clinical use.
🙋‍♂️ Author

Kyle Spengler

    📧 kyle.s.delivery@gmail.com

    🌐 LinkedIn

    💻 GitHub
