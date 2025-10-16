# 🚴‍♀️ Bike Sharing Demand Prediction using AutoGluon

## 📘 Project Overview
This project was completed as part of the **AWS AI & ML Scholarship Program**. It focuses on predicting **bike-sharing demand** using the **AutoGluon** library for automated machine learning (AutoML).

The project is based on the **[Bike Sharing Demand competition on Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand)**. The goal is to predict the number of bikes rented in each hour based on various time and weather-related features.

Accurately predicting demand helps companies like **Uber, Lyft, and DoorDash** optimize their resources, plan better for spikes in demand, and improve customer experience by minimizing wait times.

---

## 🧠 Key Objectives
- Train and optimize multiple models using **AutoGluon’s Tabular Prediction** feature.  
- Perform **feature engineering** and **exploratory data analysis (EDA)** to understand patterns in the data.  
- Experiment with **hyperparameter tuning** to improve model performance.  
- Submit predictions to Kaggle and analyze competition leaderboard results.  
- Document insights and model iterations in a final **competition report**.

---

## 📂 Dataset
The dataset was obtained from Kaggle using the **Kaggle CLI**. It includes:
- `train.csv` — training data containing features and target variable (`count`)
- `test.csv` — testing data for generating predictions
- `sampleSubmission.csv` — submission template for Kaggle

Data was loaded into **Pandas DataFrames** for preprocessing and exploration.

---

## 🔍 Feature Engineering & Data Analysis
- Created new features from existing ones (e.g., extracting date/time information).  
- Converted certain numeric columns to categorical types to enhance AutoGluon performance.  
- Visualized the distribution of features using **Matplotlib histograms**.  
- Discovered meaningful patterns during EDA that influenced feature selection and model tuning.

**Example:**
```python
for col in ['season', 'weather']:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
```

---

## ⚙️ Model Training with AutoGluon
- Used **`TabularPredictor`** from AutoGluon to automatically train multiple models.  
- Experimented with **hyperparameter tuning** to optimize model performance.  
- Evaluated models using AutoGluon’s **`fit_summary()`** and **`leaderboard()`** functions.  
- Selected the best-performing model based on validation metrics and Kaggle leaderboard scores.

**Example:**
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label="count").fit("train.csv", presets="best_quality", time_limit=3600)
```

---

## 📈 Model Evaluation & Results
- Generated predictions on the test dataset using the trained model.  
- Submitted predictions to **Kaggle** for public leaderboard scoring via the **Kaggle CLI**.  
- Tracked and visualized:  
  - Model performance metrics across iterations (from `leaderboard()`)
  - Kaggle leaderboard scores over different submissions

These results were plotted using **Matplotlib** and analyzed to understand how changes in features and hyperparameters affected outcomes.

**Example submission:**
```bash
kaggle competitions submit -c bike-sharing-demand -f submissions/submission_01.csv -m "AutoGluon run 01"
```

---

## 🏁 Competition Report
The final report included:
- Comparison of different model performances using AutoGluon’s summary outputs.  
- Insights from EDA and how new features improved prediction accuracy.  
- A table detailing each hyperparameter configuration and corresponding Kaggle score.  
- Explanation of why specific hyperparameter changes influenced the model’s results.

---

## 🧰 Tools & Technologies Used
- **Python**  
- **AutoGluon**  
- **Pandas**  
- **Matplotlib**  
- **Kaggle API**  
- **Jupyter Notebook / SageMaker Studio**

---

## 🌟 Key Learnings
- Gained practical experience using **AutoML tools** to handle real-world prediction tasks.  
- Understood the importance of **feature engineering** and **data type optimization**.  
- Learned how to iteratively improve model performance using **Kaggle submissions and evaluation metrics**.

---

## 🧩 How to Reproduce
1. Install dependencies:
   ```bash
   pip install autogluon.tabular pandas matplotlib kaggle
   ```
2. Download the dataset:
   ```bash
   kaggle competitions download -c bike-sharing-demand
   unzip bike-sharing-demand.zip -d data/
   ```
3. Run training and predictions using the provided notebooks or scripts.
4. Submit results to Kaggle using:
   ```bash
   kaggle competitions submit -c bike-sharing-demand -f submissions/submission_01.csv -m "AutoGluon run 01"
   ```

---

## 📜 Acknowledgment
This project was completed as part of the **AWS AI & ML Scholarship Program**, designed to provide hands-on experience in machine learning using modern tools and frameworks.

**Author:** Adenike
