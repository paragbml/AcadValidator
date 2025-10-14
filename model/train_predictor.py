#!/usr/bin/env python3
"""
Institutional Performance Prediction Module
AI-Based Institutional Performance Tracker

This module:
1. Loads the trained model
2. Accepts user input for institutional metrics
3. Predicts Final Performance Index
4. Provides performance classification and recommendations
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class PerformancePredictor:
    def __init__(self, model_path='predictor_model.pkl'):
        """Initialize predictor with trained model"""
        self.model_path = model_path
        self.model_package = None
        self.load_model()
        
        # Performance level definitions
        self.performance_levels = {
            'Excellent': {'min': 85, 'max': 100, 'color': 'success', 'description': 'Outstanding institutional performance'},
            'Very Good': {'min': 75, 'max': 85, 'color': 'info', 'description': 'Strong institutional performance'},
            'Good': {'min': 65, 'max': 75, 'color': 'primary', 'description': 'Satisfactory institutional performance'},
            'Average': {'min': 55, 'max': 65, 'color': 'warning', 'description': 'Room for improvement'},
            'Needs Improvement': {'min': 0, 'max': 55, 'color': 'danger', 'description': 'Requires significant enhancement'}
        }
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            self.model_package = joblib.load(self.model_path)
            print("✅ Model loaded successfully!")
            print(f"   Features: {len(self.model_package['feature_names'])}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_performance_level(self, score):
        """Classify performance score into levels"""
        for level, bounds in self.performance_levels.items():
            if bounds['min'] <= score < bounds['max']:
                return level, bounds
        return 'Excellent', self.performance_levels['Excellent']  # For scores >= 100
    
    def predict_performance(self, input_data):
        """
        Predict institutional performance based on input metrics
        
        Args:
            input_data (dict): Dictionary containing institutional metrics
            
        Returns:
            dict: Prediction results with score, level, and recommendations
        """
        if not self.model_package:
            return {'error': 'Model not loaded'}
        
        try:
            # Extract model components
            model = self.model_package['model']
            scaler = self.model_package['scaler']
            feature_names = self.model_package['feature_names']
            
            # Create feature vector
            features = self.prepare_features(input_data)
            
            # Ensure all required features are present
            feature_vector = []
            for feature in feature_names:
                if feature in features:
                    feature_vector.append(features[feature])
                else:
                    print(f"⚠️  Warning: Missing feature {feature}, using default value")
                    feature_vector.append(0)  # Default value for missing features
            
            # Convert to numpy array and reshape
            X = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            predicted_score = model.predict(X_scaled)[0]
            
            # Ensure score is within bounds
            predicted_score = max(30, min(100, predicted_score))
            
            # Get performance level
            level, level_info = self.get_performance_level(predicted_score)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(input_data, predicted_score)
            
            # Calculate feature contributions (simplified)
            feature_importance = model.feature_importances_
            contributions = {}
            for i, feature in enumerate(feature_names):
                if i < len(feature_vector):
                    contributions[feature] = feature_importance[i] * feature_vector[i]
            
            return {
                'predicted_score': round(predicted_score, 2),
                'performance_level': level,
                'level_info': level_info,
                'recommendations': recommendations,
                'input_metrics': input_data,
                'feature_contributions': contributions
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def prepare_features(self, input_data):
        """
        Prepare feature vector from input data
        
        Args:
            input_data (dict): Raw input metrics
            
        Returns:
            dict: Processed features
        """
        features = {}
        
        # Direct features
        direct_features = [
            'AICTE_Approval_Score', 'UGC_Rating', 'NIRF_Rank',
            'Placement_Percentage', 'Faculty_Student_Ratio', 'Research_Projects',
            'Infrastructure_Score', 'Student_Satisfaction_Score'
        ]
        
        for feature in direct_features:
            features[feature] = input_data.get(feature, 0)
        
        # Engineered features
        # College tier (estimate based on other metrics)
        tier_score = (
            input_data.get('AICTE_Approval_Score', 0) * 0.3 +
            input_data.get('UGC_Rating', 0) * 10 * 0.2 +
            (200 - input_data.get('NIRF_Rank', 200)) * 0.5 * 0.2 +
            input_data.get('Placement_Percentage', 0) * 0.3
        )
        features['College_Tier'] = tier_score
        
        # Years since 2019 (assume current prediction is for 2023)
        features['Years_Since_2019'] = 4  # 2023 - 2019
        
        # Placement to satisfaction ratio
        placement = input_data.get('Placement_Percentage', 1)
        satisfaction = input_data.get('Student_Satisfaction_Score', 1)
        features['Placement_to_Satisfaction_Ratio'] = placement / max(satisfaction, 1)
        
        # Faculty quality score (inverse of ratio)
        faculty_ratio = input_data.get('Faculty_Student_Ratio', 15)
        features['Faculty_Quality_Score'] = 100 / max(faculty_ratio, 1)
        
        return features
    
    def generate_recommendations(self, input_data, predicted_score):
        """
        Generate improvement recommendations based on input metrics
        
        Args:
            input_data (dict): Input institutional metrics
            predicted_score (float): Predicted performance score
            
        Returns:
            list: List of recommendation strings
        """
        recommendations = []
        
        # AICTE Score recommendations
        aicte_score = input_data.get('AICTE_Approval_Score', 0)
        if aicte_score < 70:
            recommendations.append("🎯 Improve AICTE compliance and approval scores through better documentation and process adherence")
        elif aicte_score < 85:
            recommendations.append("📈 Enhance AICTE approval metrics to reach excellence level")
        
        # UGC Rating recommendations
        ugc_rating = input_data.get('UGC_Rating', 0)
        if ugc_rating < 6:
            recommendations.append("🏛️ Focus on UGC quality parameters including faculty qualifications and research output")
        elif ugc_rating < 8:
            recommendations.append("📚 Strengthen academic programs to improve UGC ratings")
        
        # NIRF Rank recommendations
        nirf_rank = input_data.get('NIRF_Rank', 200)
        if nirf_rank > 100:
            recommendations.append("🏆 Work on NIRF ranking parameters: teaching, research, outreach, and perception")
        elif nirf_rank > 50:
            recommendations.append("🥇 Enhance research publications and industry collaborations for better NIRF ranking")
        
        # Placement recommendations
        placement = input_data.get('Placement_Percentage', 0)
        if placement < 60:
            recommendations.append("💼 Strengthen placement cell activities and industry partnerships")
        elif placement < 80:
            recommendations.append("🤝 Expand corporate relations and skill development programs")
        
        # Faculty ratio recommendations
        faculty_ratio = input_data.get('Faculty_Student_Ratio', 15)
        if faculty_ratio > 20:
            recommendations.append("👨‍🏫 Improve faculty-student ratio by hiring more qualified faculty members")
        elif faculty_ratio > 15:
            recommendations.append("📖 Consider optimizing class sizes and faculty allocation")
        
        # Research recommendations
        research_projects = input_data.get('Research_Projects', 0)
        if research_projects < 10:
            recommendations.append("🔬 Increase research activities and project funding")
        elif research_projects < 20:
            recommendations.append("📝 Encourage faculty to apply for more research grants")
        
        # Infrastructure recommendations
        infrastructure = input_data.get('Infrastructure_Score', 0)
        if infrastructure < 70:
            recommendations.append("🏗️ Invest in infrastructure development and maintenance")
        elif infrastructure < 85:
            recommendations.append("🔧 Upgrade existing facilities and add modern amenities")
        
        # Student satisfaction recommendations
        satisfaction = input_data.get('Student_Satisfaction_Score', 0)
        if satisfaction < 70:
            recommendations.append("😊 Focus on student services, campus life, and feedback mechanisms")
        elif satisfaction < 85:
            recommendations.append("🎓 Enhance student support services and extracurricular activities")
        
        # Overall recommendations based on predicted score
        if predicted_score < 60:
            recommendations.append("🚀 Implement comprehensive institutional development plan")
            recommendations.append("📊 Establish performance monitoring and quality assurance systems")
        elif predicted_score < 75:
            recommendations.append("⭐ Focus on specific weak areas to achieve higher performance tier")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def interactive_prediction(self):
        """Interactive command-line prediction interface"""
        print("🎯 AI-Based Institutional Performance Predictor")
        print("=" * 60)
        print("Please enter the following institutional metrics:")
        print()
        
        try:
            # Collect input data
            input_data = {}
            
            print("📊 Basic Metrics:")
            input_data['AICTE_Approval_Score'] = float(input("AICTE Approval Score (40-100): "))
            input_data['UGC_Rating'] = float(input("UGC Rating (3-10): "))
            input_data['NIRF_Rank'] = int(input("NIRF Rank (1-200): "))
            
            print("\n👥 Human Resources:")
            input_data['Placement_Percentage'] = float(input("Placement Percentage (25-100): "))
            input_data['Faculty_Student_Ratio'] = float(input("Faculty-Student Ratio (8-25): "))
            
            print("\n🔬 Research & Infrastructure:")
            input_data['Research_Projects'] = int(input("Number of Research Projects (0-50): "))
            input_data['Infrastructure_Score'] = float(input("Infrastructure Score (40-100): "))
            input_data['Student_Satisfaction_Score'] = float(input("Student Satisfaction Score (40-100): "))
            
            # Make prediction
            print("\n🤖 Analyzing institutional performance...")
            result = self.predict_performance(input_data)
            
            if 'error' in result:
                print(f"❌ {result['error']}")
                return
            
            # Display results
            print("\n" + "=" * 60)
            print("🎉 PREDICTION RESULTS")
            print("=" * 60)
            
            print(f"📈 Predicted Performance Index: {result['predicted_score']}")
            print(f"🏆 Performance Level: {result['performance_level']}")
            print(f"📝 Description: {result['level_info']['description']}")
            
            # Performance interpretation
            score = result['predicted_score']
            if score >= 85:
                interpretation = "🌟 Excellent! Your institution shows outstanding performance across all metrics."
            elif score >= 75:
                interpretation = "✨ Very Good! Strong performance with minor areas for enhancement."
            elif score >= 65:
                interpretation = "👍 Good! Solid performance with room for targeted improvements."
            elif score >= 55:
                interpretation = "⚠️  Average performance. Focus on key improvement areas."
            else:
                interpretation = "🔧 Significant improvement needed across multiple areas."
            
            print(f"\n💡 {interpretation}")
            
            # Display recommendations
            print(f"\n🎯 IMPROVEMENT RECOMMENDATIONS:")
            print("-" * 40)
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec}")
            
            print(f"\n📊 Thank you for using the Institutional Performance Predictor!")
            
            return result
            
        except KeyboardInterrupt:
            print("\n\n👋 Prediction cancelled by user.")
        except Exception as e:
            print(f"\n❌ Error during prediction: {e}")

def main():
    """Main function for standalone prediction"""
    predictor = PerformancePredictor()
    
    if predictor.model_package:
        predictor.interactive_prediction()
    else:
        print("❌ Please run model_train.py first to train the model!")

if __name__ == "__main__":
    main()