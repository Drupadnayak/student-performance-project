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

---

## ⚙️ Customization Options

### Change Pass/Fail Threshold
In `app.py` (Line ~80):
```python
# Change from 40 to desired threshold
status = "PASS" if predicted_score >= 40 else "FAIL"
```

### Adjust Feature Ranges
In `templates/index.html`, modify input attributes:
```html
<!-- Example: Change study hours max to 20 -->
<input min="0" max="20" ...>
```

### Modify Model Features
To add more features to the model:
1. Update `train_model.py` (add new columns)
2. Update `app.py` (modify prediction function)
3. Update `index.html` (add input fields)
4. Retrain the model

---

## 🐛 Troubleshooting

### Issue: "Model not found"
**Solution:** Make sure you've run `python train_model.py` first
```bash
python train_model.py
```

### Issue: "Port 5000 already in use"
**Solution:** Use a different port in `app.py`:
```python
app.run(port=5001)
```

### Issue: "ModuleNotFoundError"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Predictions seem wrong
**Solution:** 
- Check input values are within expected ranges
- Retrain model with larger/better dataset
- Verify training data quality

---

## 📚 Viva Questions & Answers

See the comprehensive document provided for 12+ viva questions with detailed answers.

**Key Topics:**
- Why Linear Regression?
- What is R² Score?
- Data preprocessing steps
- Train-test split rationale
- Model generalization
- Feature scaling importance
- Error handling
- Bias and fairness
- Future improvements

---

## 🎓 For Academic Submission

This project includes:

✅ **Complete ML Pipeline**
- Data collection/generation
- Preprocessing
- Training
- Evaluation
- Deployment

✅ **Full-Stack Application**
- Backend (Flask API)
- Frontend (HTML/CSS)
- Model integration

✅ **Documentation**
- README (this file)
- Code comments
- Viva Q&A
- PPT structure

✅ **Production-Ready**
- Error handling
- Input validation
- Performance metrics
- Real-time predictions

---

## 📝 Project Report Sections

Refer to the separate document for:
1. Introduction & Problem Statement
2. Methodology & Data Preprocessing
3. Model Selection & Training
4. Results & Evaluation
5. System Architecture
6. Limitations & Future Scope

---

## 🔐 Notes on Deployment

**For Local Development:**
- Current setup runs on `http://localhost:5000`
- Debug mode is enabled for development

**For Production Deployment:**
1. Set `debug=False` in `app.py`
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Add authentication if needed
4. Set up proper error logging
5. Deploy on cloud (Heroku, AWS, Azure, etc.)

---

## 📞 Support & Help

For questions or issues:
1. Check the troubleshooting section
2. Review code comments in `train_model.py` and `app.py`
3. Verify input data format
4. Check console output for error messages

---

## 📜 License

This is an academic project for educational purposes.

---

## ✨ Features Checklist

- ✅ Linear Regression Model
- ✅ Data Preprocessing (cleaning, scaling)
- ✅ Model Evaluation (R² Score, MAE, RMSE)
- ✅ Flask Backend API
- ✅ HTML/CSS Frontend
- ✅ Real-time Predictions
- ✅ Pass/Fail Classification
- ✅ Input Validation
- ✅ Error Handling
- ✅ Model Serialization (Pickle)
- ✅ Responsive Design
- ✅ Sample Data Generation
- ✅ Comprehensive Documentation

---

**Last Updated:** April 2026
**Status:** ✅ Ready for Academic Submission
