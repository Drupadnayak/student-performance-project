"""
Student Performance Prediction System - Model Training Script
Machine Learning Model: Linear Regression
Author: [Your Name]
Date: April 2026
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os

# ============================================================================
# STEP 1: GENERATE OR LOAD DATASET
# ============================================================================

def generate_sample_dataset(n_samples=200):
    """
    Generates a realistic sample dataset for training.
    You can replace this with your actual CSV data.
    
    Parameters:
    - n_samples: Number of student records
    
    Returns:
    - DataFrame with features and target
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic data
    study_hours = np.random.uniform(2, 15, n_samples)
    attendance = np.random.uniform(40, 100, n_samples)
    previous_score = np.random.uniform(30, 95, n_samples)
    
    # Generate final score with some relationship to features
    # Final Score = 0.5 * Study + 0.3 * Attendance + 0.15 * Previous + noise
    final_score = (
        0.5 * study_hours + 
        0.3 * attendance + 
        0.15 * previous_score + 
        np.random.normal(0, 5, n_samples)  # Add realistic noise
    )
    
    # Clip scores to realistic range [0, 100]
    final_score = np.clip(final_score, 0, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        'study_hours': study_hours,
        'attendance': attendance,
        'previous_score': previous_score,
        'final_score': final_score
    })
    
    return data


def load_custom_dataset(csv_path):
    """
    Load your own CSV dataset.
    CSV should have columns: study_hours, attendance, previous_score, final_score
    """
    data = pd.read_csv(csv_path)
    print(f"✓ Loaded dataset from {csv_path}")
    print(f"  Shape: {data.shape}")
    return data


# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================

def preprocess_data(data):
    """
    Preprocess the dataset:
    - Handle missing values
    - Remove outliers (IQR method)
    - Prepare features and target
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print(f"⚠ Found missing values:\n{missing_values}")
        data = data.dropna()
        print(f"✓ Dropped {missing_values.sum()} rows with missing values")
    
    # Remove outliers using IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    outlier_mask = ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
    initial_size = len(data)
    data = data[outlier_mask]
    removed = initial_size - len(data)
    
    if removed > 0:
        print(f"⚠ Found {removed} outliers")
        print(f"✓ Removed outliers using IQR method")
    
    print(f"✓ Dataset shape after preprocessing: {data.shape}")
    print(f"\nDataset Statistics:")
    print(data.describe())
    
    return data


def scale_features(X_train, X_test=None):
    """
    Scale features to 0-1 range using MinMaxScaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    80% training, 20% testing
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n✓ Training set size: {len(X_train)} samples ({100*(1-test_size):.0f}%)")
    print(f"✓ Testing set size: {len(X_test)} samples ({100*test_size:.0f}%)")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# STEP 4: TRAIN LINEAR REGRESSION MODEL
# ============================================================================

def train_model(X_train, y_train):
    """
    Train Linear Regression model
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✓ Linear Regression model trained successfully")
    print(f"\nModel Coefficients:")
    features = ['Study Hours', 'Attendance %', 'Previous Score']
    for feature, coef in zip(features, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept (Base Value): {model.intercept_:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  Final Score = {model.coef_[0]:.4f}×Study + {model.coef_[1]:.4f}×Attendance + {model.coef_[2]:.4f}×Previous + {model.intercept_:.4f}")
    
    return model


# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on training and testing data
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # R² Score
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    # Mean Absolute Error
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    # Root Mean Squared Error
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Display results
    print("\n📊 TRAINING SET METRICS:")
    print(f"  R² Score: {r2_train:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_train:.2f} marks")
    print(f"  Root Mean Squared Error (RMSE): {rmse_train:.2f} marks")
    
    print("\n📊 TESTING SET METRICS:")
    print(f"  R² Score: {r2_test:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_test:.2f} marks")
    print(f"  Root Mean Squared Error (RMSE): {rmse_test:.2f} marks")
    
    print("\n" + "-"*60)
    print(f"✓ Model explains {r2_test*100:.2f}% of variance in student scores")
    
    if r2_test > 0.7:
        print("✓ Model Performance: GOOD ✓")
    elif r2_test > 0.5:
        print("⚠ Model Performance: MODERATE")
    else:
        print("⚠ Model Performance: NEEDS IMPROVEMENT")
    print("-"*60)
    
    return {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test
    }


# ============================================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================================

def make_predictions(model, scaler, sample_data):
    """
    Make predictions on new data
    sample_data: DataFrame with columns [study_hours, attendance, previous_score]
    """
    scaled_data = scaler.transform(sample_data)
    predictions = model.predict(scaled_data)
    
    results = pd.DataFrame({
        'Study Hours': sample_data['study_hours'],
        'Attendance %': sample_data['attendance'],
        'Previous Score': sample_data['previous_score'],
        'Predicted Final Score': predictions,
        'Status': ['PASS' if score >= 40 else 'FAIL' for score in predictions]
    })
    
    return results


# ============================================================================
# STEP 7: SAVE MODEL & SCALER
# ============================================================================

def save_model(model, scaler, output_dir='models'):
    """
    Save trained model and scaler as pickle files
    """
    print("\n" + "="*60)
    print("SAVING MODEL & SCALER")
    print("="*60)
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created directory: {output_dir}")
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    return model_path, scaler_path


# ============================================================================
# STEP 8: LOAD SAVED MODEL
# ============================================================================

def load_model(model_path, scaler_path):
    """
    Load saved model and scaler from pickle files
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Scaler loaded from: {scaler_path}")
    
    return model, scaler


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("Machine Learning Model Training Pipeline")
    print("="*60)
    
    # Step 1: Load or Generate Dataset
    print("\n[1/7] Loading Dataset...")
    # Option A: Generate sample data (uncomment to use)
    data = generate_sample_dataset(n_samples=200)
    
    # Option B: Load your own CSV (uncomment to use)
    # data = load_custom_dataset('path_to_your_data.csv')
    
    # Step 2: Preprocess Data
    print("\n[2/7] Preprocessing Data...")
    data = preprocess_data(data)
    
    # Step 3: Prepare Features and Target
    print("\n[3/7] Preparing Features and Target...")
    X = data[['study_hours', 'attendance', 'previous_score']]
    y = data['final_score']
    
    # Step 4: Split Data
    print("\n[4/7] Splitting Data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Step 5: Scale Features
    print("\n[5/7] Scaling Features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("✓ Features scaled to [0, 1] range")
    
    # Step 6: Train Model
    print("\n[6/7] Training Model...")
    model = train_model(X_train_scaled, y_train)
    
    # Step 7: Evaluate Model
    print("\n[7/7] Evaluating Model...")
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save Model
    print("\n[8/7] Saving Model & Scaler...")
    model_path, scaler_path = save_model(model, scaler, output_dir='models')
    
    # Test Predictions
    print("\n" + "="*60)
    print("TEST PREDICTIONS")
    print("="*60)
    
    test_samples = pd.DataFrame({
        'study_hours': [10, 5, 12, 8],
        'attendance': [85, 60, 90, 70],
        'previous_score': [75, 50, 80, 65]
    })
    
    predictions = make_predictions(model, scaler, test_samples)
    print("\nSample Predictions:")
    print(predictions.to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Model saved in 'models/' directory")
    print(f"   - model.pkl: Trained Linear Regression model")
    print(f"   - scaler.pkl: Feature scaler for preprocessing")
    print(f"\n💡 Next Step: Integrate this model with Flask application")
    print("="*60 + "\n")
