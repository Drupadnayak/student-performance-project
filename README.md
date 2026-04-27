<img width="1920" height="2267" alt="screencapture-localhost-5000-2026-04-27-23_18_01" src="https://github.com/user-attachments/assets/a55d9f5f-4f43-4972-ba84-79d2fe5c2fd4" />

# Student Performance Prediction System

A complete Machine Learning project that predicts student final examination scores using Linear Regression.

## 📋 Project Overview

**Objective:** Build a predictive system that forecasts a student's final score based on three key factors:
- Study Hours per week
- Attendance Percentage
- Previous Exam Score

**Technology Stack:**
- **Backend:** Flask (Python)
- **Frontend:** HTML5 + CSS3
- **ML Model:** Linear Regression (scikit-learn)
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Pickle

---

## 📁 Project Structure

```
student-performance-project/
│
├── train_model.py              # Model training script
├── app.py                      # Flask backend application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── templates/
│   └── index.html             # Frontend HTML form & UI
│
├── static/
│   └── style.css              # CSS stylesheet
│
└── models/                     # (Generated after training)
    ├── model.pkl              # Trained Linear Regression model
    └── scaler.pkl             # Feature scaler for preprocessing
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd student-performance-project

# Install required Python packages
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python train_model.py
```

**What happens:**
- Generates sample dataset (200 students) or loads your CSV
- Cleans and preprocesses data
- Trains Linear Regression model
- Evaluates model performance (R² Score, MAE, RMSE)
- Saves model and scaler to `models/` folder

**Expected Output:**
```
============================================================
STUDENT PERFORMANCE PREDICTION SYSTEM
Machine Learning Model Training Pipeline
============================================================

[1/7] Loading Dataset...
✓ Dataset shape: (200, 4)

[2/7] Preprocessing Data...
✓ Dataset shape after preprocessing: (200, 4)

...

[7/7] Evaluating Model...
📊 TESTING SET METRICS:
  R² Score: 0.7800
  Mean Absolute Error (MAE): 3.45 marks
  Root Mean Squared Error (RMSE): 4.12 marks

✓ Model explains 78.00% of variance in student scores

============================================================
✅ MODEL TRAINING COMPLETE!
============================================================
```

### Step 3: Run Flask Application

```bash
python app.py
```

**Console Output:**
```
============================================================
STUDENT PERFORMANCE PREDICTION SYSTEM
Flask Backend Server
============================================================

✓ Model and scaler loaded successfully

✓ All systems ready!
🚀 Starting Flask server...

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5000
```

### Step 4: Access the Web Application

Open your browser and go to:
```
http://localhost:5000
```

---

## 💡 How to Use the System

### Web Interface

1. **Enter Study Hours:** Input hours studied per week (0-24)
2. **Enter Attendance %:** Input attendance percentage (0-100)
3. **Enter Previous Score:** Input previous exam score (0-100)
4. **Click "Predict Score":** Get instant prediction

### Sample Predictions

| Study Hours | Attendance % | Previous Score | Predicted Score | Status |
|------------|-------------|----------------|-----------------|--------|
| 10 | 85 | 75 | ~78 | PASS ✅ |
| 5 | 60 | 50 | ~45 | PASS ✅ |
| 3 | 50 | 40 | ~35 | FAIL ❌ |
| 12 | 90 | 80 | ~82 | PASS ✅ |

---

## 🎯 Model Details

### Features Used
- **Study Hours:** Hours studied per week (0-24)
- **Attendance %:** Class attendance percentage (0-100)
- **Previous Score:** Score from previous exam (0-100)

### Target Variable
- **Final Score:** Student's final examination score (0-100)

### Model Type
- **Algorithm:** Linear Regression (Ordinary Least Squares)
- **Equation:** 
  ```
  Final Score = 0.5×Study + 0.3×Attendance + 0.15×Previous + Constant
  ```

### Performance Metrics
- **R² Score:** ~0.78 (Model explains 78% of variance)
- **MAE:** ±3-4 marks on average
- **RMSE:** ~4-5 marks
- **Pass Threshold:** ≥ 40 marks = PASS

---

## 📊 API Endpoints

### 1. Home Page
```
GET http://localhost:5000/
```
Returns the web interface (HTML form)

### 2. Make Prediction
```
POST http://localhost:5000/predict
Content-Type: application/json

{
  "study_hours": 10,
  "attendance": 85,
  "previous_score": 75
}
```

**Response:**
```json
{
  "predicted_score": 78.45,
  "status": "PASS",
  "confidence": 78.45,
  "feedback": "Excellent performance expected! Keep up the great work!"
}
```

### 3. Model Information
```
GET http://localhost:5000/info
```

**Response:**
```json
{
  "model_name": "Linear Regression",
  "features": ["Study Hours (0-24)", "Attendance % (0-100)", "Previous Score (0-100)"],
  "target": "Final Score (0-100)",
  "pass_threshold": 40,
  "expected_accuracy": "R² Score ~0.78"
}
```

### 4. Health Check
```
GET http://localhost:5000/health
```

---

## 🔧 Using Your Own Data

To train the model with your own CSV dataset:

1. **Create a CSV file** with columns:
   ```
   study_hours,attendance,previous_score,final_score
   8,85,75,78
   5,60,50,45
   ...
   ```

2. **Update `train_model.py`** (Line ~105):
   ```python
   # Replace this:
   data = generate_sample_dataset(n_samples=200)
   
   # With this:
   data = load_custom_dataset('path_to_your_data.csv')
   ```

3. **Run training:**
   ```bash
   python train_model.py
   ```

---

## 📈 Model Training Process

### Step-by-Step Workflow

1. **Data Loading** → Load CSV or generate sample data
2. **Data Cleaning** → Remove missing values and outliers
3. **Preprocessing** → Feature scaling (MinMaxScaler)
4. **Train-Test Split** → 80% training, 20% testing
5. **Model Training** → Fit Linear Regression on training data
6. **Evaluation** → Calculate R², MAE, RMSE on test data
7. **Model Saving** → Pickle model and scaler for deployment

--
