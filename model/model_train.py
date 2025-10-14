#!/usr/bin/env python3
"""
Machine Learning Model Training Module
AI-Based Institutional Performance Tracker

This module:
1. Loads the synthetic dataset
2. Performs data preprocessing and feature engineering
3. Trains a RandomForestRegressor model
4. Evaluates model performance
5. Saves the trained model for prediction use
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class InstitutionalPerformanceModel:
    def __init__(self, data_path='../data/ugc_aicte_synthetic_dataset_2019_2023.csv'):
        """Initialize the model with dataset path"""
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_name = 'Final_Performance_Index'
        
    def load_data(self):
        """Load and examine the dataset"""
        print("📊 Loading dataset...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
            print(f"   Date range: {self.df['Year'].min()} - {self.df['Year'].max()}")
            print(f"   Unique institutions: {self.df['College_Name'].nunique()}")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n🔍 Exploratory Data Analysis")
        print("=" * 50)
        
        # Basic statistics
        print("\n📈 Target Variable Statistics:")
        print(self.df[self.target_name].describe())
        
        # Missing values check
        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing[missing > 0])
        
        # Correlation analysis
        print("\n📊 Feature Correlations with Target:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = self.df[numeric_cols].corr()[self.target_name].sort_values(ascending=False)
        print(correlations.drop(self.target_name))
        
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("\n🔧 Preprocessing data...")
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Feature Engineering
        # 1. Encode college names (create college tier feature)
        college_performance = df_processed.groupby('College_Name')[self.target_name].mean()
        df_processed['College_Tier'] = df_processed['College_Name'].map(college_performance)
        
        # 2. Create time-based features
        df_processed['Years_Since_2019'] = df_processed['Year'] - 2019
        
        # 3. Create ratio features
        df_processed['Placement_to_Satisfaction_Ratio'] = (
            df_processed['Placement_Percentage'] / df_processed['Student_Satisfaction_Score']
        )
        
        # 4. Create performance categories
        df_processed['AICTE_Category'] = pd.cut(df_processed['AICTE_Approval_Score'], 
                                               bins=[0, 60, 80, 100], 
                                               labels=['Low', 'Medium', 'High'])
        
        # 5. Normalize Faculty Ratio (inverse relationship)
        df_processed['Faculty_Quality_Score'] = 100 / df_processed['Faculty_Student_Ratio']
        
        # Select features for training
        feature_cols = [
            'AICTE_Approval_Score', 'UGC_Rating', 'NIRF_Rank', 
            'Placement_Percentage', 'Faculty_Student_Ratio', 'Research_Projects',
            'Infrastructure_Score', 'Student_Satisfaction_Score',
            'College_Tier', 'Years_Since_2019', 'Placement_to_Satisfaction_Ratio',
            'Faculty_Quality_Score'
        ]
        
        # Encode categorical features if any
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in feature_cols:
                df_processed[col] = self.label_encoder.fit_transform(df_processed[col])
        
        # Prepare feature matrix and target
        self.X = df_processed[feature_cols]
        self.y = df_processed[self.target_name]
        self.feature_names = feature_cols
        
        print(f"✅ Features prepared: {len(feature_cols)} features")
        print(f"   Feature names: {feature_cols}")
        
        return df_processed
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the Random Forest model"""
        print("\n🤖 Training Random Forest Model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Scale features (optional for Random Forest, but good practice)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Store test data for evaluation
        self.X_test, self.y_test, self.y_test_pred = X_test_scaled, y_test, y_test_pred
        
        print("✅ Model training completed!")
        return y_train_pred, y_test_pred
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n📊 Model Evaluation")
        print("=" * 50)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, self.y_test_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        r2 = r2_score(self.y_test, self.y_test_pred)
        
        print(f"📈 Performance Metrics:")
        print(f"   Mean Absolute Error (MAE): {mae:.2f}")
        print(f"   Root Mean Square Error (RMSE): {rmse:.2f}")
        print(f"   R² Score: {r2:.4f}")
        
        # Performance interpretation
        if r2 > 0.9:
            performance = "Excellent"
        elif r2 > 0.8:
            performance = "Very Good"
        elif r2 > 0.7:
            performance = "Good"
        elif r2 > 0.6:
            performance = "Fair"
        else:
            performance = "Needs Improvement"
        
        print(f"   Model Performance: {performance}")
        
        # Cross-validation
        print(f"\n🔄 Cross-Validation Results:")
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring='r2')
        print(f"   CV R² Scores: {cv_scores}")
        print(f"   Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        print(f"\n🎯 Top 10 Feature Importances:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance
        }
    
    def save_model(self, model_path='predictor_model.pkl'):
        """Save the trained model and preprocessing objects"""
        print(f"\n💾 Saving model to {model_path}...")
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }
        
        try:
            joblib.dump(model_package, model_path)
            print("✅ Model saved successfully!")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def create_visualizations(self):
        """Create and save training visualizations"""
        print("\n📊 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(self.y_test, self.y_test_pred, alpha=0.6)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Performance Index')
        axes[0, 0].set_ylabel('Predicted Performance Index')
        axes[0, 0].set_title('Actual vs Predicted Performance')
        
        # 2. Residuals plot
        residuals = self.y_test - self.y_test_pred
        axes[0, 1].scatter(self.y_test_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Performance Index')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # 3. Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        axes[1, 0].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[1, 0].set_yticks(range(len(feature_importance)))
        axes[1, 0].set_yticklabels(feature_importance['feature'])
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Feature Importance Rankings')
        
        # 4. Performance distribution
        axes[1, 1].hist(self.y_test, alpha=0.5, label='Actual', bins=20)
        axes[1, 1].hist(self.y_test_pred, alpha=0.5, label='Predicted', bins=20)
        axes[1, 1].set_xlabel('Performance Index')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Performance Index Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved!")

def main():
    """Main training pipeline"""
    print("🎯 AI-Based Institutional Performance Tracker")
    print("🤖 Machine Learning Model Training")
    print("=" * 60)
    
    # Initialize model
    model = InstitutionalPerformanceModel()
    
    # Load and explore data
    if not model.load_data():
        return
    
    model.explore_data()
    
    # Preprocess data
    model.preprocess_data()
    
    # Train model
    model.train_model()
    
    # Evaluate model
    results = model.evaluate_model()
    
    # Save model
    model.save_model()
    
    # Create visualizations
    model.create_visualizations()
    
    print("\n🎉 Training pipeline completed successfully!")
    print("📁 Model saved as 'predictor_model.pkl'")
    print("📊 Evaluation plots saved as 'model_evaluation_plots.png'")

if __name__ == "__main__":
    main()