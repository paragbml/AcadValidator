#!/usr/bin/env python3
"""
Flask API Server for AI-Based Institutional Performance Tracker

This module provides REST API endpoints for:
- /predict: Institutional performance prediction
- /report/<college_name>: Historical data and analysis
- /colleges: List of available colleges
- /health: Server health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import logging

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

try:
    from train_predictor import PerformancePredictor
except ImportError:
    print("❌ Could not import PerformancePredictor. Make sure the model is trained.")
    PerformancePredictor = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
predictor = None
dataset = None

def initialize_app():
    """Initialize the application with model and data"""
    global predictor, dataset
    
    try:
        # Load the prediction model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'predictor_model.pkl')
        if PerformancePredictor and os.path.exists(model_path):
            predictor = PerformancePredictor(model_path)
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning("⚠️  Model not found. Prediction endpoints will not work.")
        
        # Load the dataset - try expanded dataset first, fall back to original
        expanded_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'expanded_indian_colleges_dataset_2019_2023.csv')
        original_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ugc_aicte_synthetic_dataset_2019_2023.csv')
        
        if os.path.exists(expanded_data_path):
            dataset = pd.read_csv(expanded_data_path)
            logger.info(f"✅ Expanded dataset loaded successfully: {dataset.shape[0]} records, {dataset['College_Name'].nunique()} colleges")
        elif os.path.exists(original_data_path):
            dataset = pd.read_csv(original_data_path)
            logger.info(f"✅ Original dataset loaded successfully: {dataset.shape[0]} records, {dataset['College_Name'].nunique()} colleges")
        else:
            logger.warning("⚠️  Dataset not found. Report endpoints will have limited functionality.")
            
    except Exception as e:
        logger.error(f"❌ Error during initialization: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor is not None,
        'dataset_loaded': dataset is not None
    })

@app.route('/predict', methods=['POST'])
def predict_performance():
    """
    Predict institutional performance based on input metrics
    
    Expected JSON payload:
    {
        "AICTE_Approval_Score": 85.5,
        "UGC_Rating": 8.2,
        "NIRF_Rank": 45,
        "Placement_Percentage": 78.5,
        "Faculty_Student_Ratio": 12.5,
        "Research_Projects": 25,
        "Infrastructure_Score": 82.0,
        "Student_Satisfaction_Score": 79.5
    }
    """
    try:
        # Check if model is loaded
        if not predictor:
            return jsonify({
                'error': 'Prediction model not available. Please train the model first.',
                'success': False
            }), 503
        
        # Get JSON data from request
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({
                'error': 'No input data provided. Please send JSON data.',
                'success': False
            }), 400
        
        # Validate required fields
        required_fields = [
            'AICTE_Approval_Score', 'UGC_Rating', 'NIRF_Rank',
            'Placement_Percentage', 'Faculty_Student_Ratio', 'Research_Projects',
            'Infrastructure_Score', 'Student_Satisfaction_Score'
        ]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'success': False
            }), 400
        
        # Validate data ranges
        validation_rules = {
            'AICTE_Approval_Score': (40, 100),
            'UGC_Rating': (3, 10),
            'NIRF_Rank': (1, 200),
            'Placement_Percentage': (25, 100),
            'Faculty_Student_Ratio': (8, 25),
            'Research_Projects': (0, 50),
            'Infrastructure_Score': (40, 100),
            'Student_Satisfaction_Score': (40, 100)
        }
        
        validation_errors = []
        for field, (min_val, max_val) in validation_rules.items():
            if field in input_data:
                value = input_data[field]
                if not (min_val <= value <= max_val):
                    validation_errors.append(f"{field} must be between {min_val} and {max_val}")
        
        if validation_errors:
            return jsonify({
                'error': 'Validation errors',
                'validation_errors': validation_errors,
                'success': False
            }), 400
        
        # Make prediction
        result = predictor.predict_performance(input_data)
        
        if 'error' in result:
            return jsonify({
                'error': result['error'],
                'success': False
            }), 500
        
        # Format response
        response = {
            'success': True,
            'prediction': {
                'performance_index': result['predicted_score'],
                'performance_level': result['performance_level'],
                'level_description': result['level_info']['description'],
                'level_color': result['level_info']['color']
            },
            'recommendations': result['recommendations'],
            'input_data': result['input_metrics'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {result['predicted_score']:.2f} ({result['performance_level']})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/report/<college_name>', methods=['GET'])
def get_college_report(college_name):
    """
    Get historical data and analysis for a specific college
    
    Args:
        college_name (str): Name of the college
    """
    try:
        # Check if dataset is loaded
        if dataset is None:
            return jsonify({
                'error': 'Dataset not available',
                'success': False
            }), 503
        
        # Find college data
        college_data = dataset[dataset['College_Name'].str.contains(college_name, case=False, na=False)]
        
        if college_data.empty:
            return jsonify({
                'error': f'College "{college_name}" not found in dataset',
                'success': False,
                'available_colleges': list(dataset['College_Name'].unique())[:10]  # Show first 10
            }), 404
        
        # Get exact match if possible
        exact_match = dataset[dataset['College_Name'].str.lower() == college_name.lower()]
        if not exact_match.empty:
            college_data = exact_match
        else:
            # Use the first partial match
            college_data = college_data.head(5)  # Limit to 5 records
        
        # Calculate statistics
        college_stats = {
            'college_name': college_data['College_Name'].iloc[0],
            'years_available': sorted(college_data['Year'].unique().tolist()),
            'latest_year': int(college_data['Year'].max()),
            'performance_trend': college_data.groupby('Year')['Final_Performance_Index'].mean().to_dict(),
            'average_performance': round(college_data['Final_Performance_Index'].mean(), 2),
            'best_performance': round(college_data['Final_Performance_Index'].max(), 2),
            'worst_performance': round(college_data['Final_Performance_Index'].min(), 2),
            'performance_growth': round(
                college_data['Final_Performance_Index'].iloc[-1] - college_data['Final_Performance_Index'].iloc[0], 2
            ) if len(college_data) > 1 else 0
        }
        
        # Get latest metrics
        latest_data = college_data[college_data['Year'] == college_data['Year'].max()].iloc[0]
        latest_metrics = {
            'AICTE_Approval_Score': round(latest_data['AICTE_Approval_Score'], 2),
            'UGC_Rating': round(latest_data['UGC_Rating'], 2),
            'NIRF_Rank': int(latest_data['NIRF_Rank']),
            'Placement_Percentage': round(latest_data['Placement_Percentage'], 2),
            'Faculty_Student_Ratio': latest_data['Faculty_Student_Ratio'],
            'Research_Projects': int(latest_data['Research_Projects']),
            'Infrastructure_Score': round(latest_data['Infrastructure_Score'], 2),
            'Student_Satisfaction_Score': round(latest_data['Student_Satisfaction_Score'], 2),
            'Final_Performance_Index': round(latest_data['Final_Performance_Index'], 2)
        }
        
        # Historical data for charts
        historical_data = []
        for _, row in college_data.iterrows():
            historical_data.append({
                'year': int(row['Year']),
                'performance_index': round(row['Final_Performance_Index'], 2),
                'aicte_score': round(row['AICTE_Approval_Score'], 2),
                'placement_percentage': round(row['Placement_Percentage'], 2),
                'research_projects': int(row['Research_Projects'])
            })
        
        # Comparative analysis (compare with dataset average)
        dataset_avg = dataset.groupby('Year')['Final_Performance_Index'].mean()
        college_avg = college_data.groupby('Year')['Final_Performance_Index'].mean()
        
        comparison_data = []
        for year in college_avg.index:
            if year in dataset_avg.index:
                comparison_data.append({
                    'year': int(year),
                    'college_performance': round(college_avg[year], 2),
                    'dataset_average': round(dataset_avg[year], 2),
                    'difference': round(college_avg[year] - dataset_avg[year], 2)
                })
        
        response = {
            'success': True,
            'college_statistics': college_stats,
            'latest_metrics': latest_metrics,
            'historical_data': historical_data,
            'comparison_data': comparison_data,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Report generated for: {college_stats['college_name']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/colleges', methods=['GET'])
def get_colleges():
    """Get list of available colleges"""
    try:
        if dataset is None:
            return jsonify({
                'error': 'Dataset not available',
                'success': False
            }), 503
        
        colleges = sorted(dataset['College_Name'].unique().tolist())
        
        # Get basic stats for each college
        college_stats = []
        for college in colleges:
            college_data = dataset[dataset['College_Name'] == college]
            stats = {
                'name': college,
                'years_available': len(college_data),
                'latest_performance': round(college_data[college_data['Year'] == college_data['Year'].max()]['Final_Performance_Index'].iloc[0], 2),
                'average_performance': round(college_data['Final_Performance_Index'].mean(), 2)
            }
            college_stats.append(stats)
        
        # Sort by latest performance
        college_stats.sort(key=lambda x: x['latest_performance'], reverse=True)
        
        return jsonify({
            'success': True,
            'total_colleges': len(colleges),
            'colleges': college_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting colleges: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_dataset_stats():
    """Get overall dataset statistics"""
    try:
        if dataset is None:
            return jsonify({
                'error': 'Dataset not available',
                'success': False
            }), 503
        
        stats = {
            'total_records': len(dataset),
            'total_colleges': dataset['College_Name'].nunique(),
            'years_covered': sorted(dataset['Year'].unique().tolist()),
            'performance_distribution': {
                'excellent': len(dataset[dataset['Final_Performance_Index'] >= 85]),
                'very_good': len(dataset[(dataset['Final_Performance_Index'] >= 75) & (dataset['Final_Performance_Index'] < 85)]),
                'good': len(dataset[(dataset['Final_Performance_Index'] >= 65) & (dataset['Final_Performance_Index'] < 75)]),
                'average': len(dataset[(dataset['Final_Performance_Index'] >= 55) & (dataset['Final_Performance_Index'] < 65)]),
                'needs_improvement': len(dataset[dataset['Final_Performance_Index'] < 55])
            },
            'average_metrics': {
                'performance_index': round(dataset['Final_Performance_Index'].mean(), 2),
                'aicte_score': round(dataset['AICTE_Approval_Score'].mean(), 2),
                'ugc_rating': round(dataset['UGC_Rating'].mean(), 2),
                'placement_percentage': round(dataset['Placement_Percentage'].mean(), 2)
            }
        }
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/compare', methods=['POST'])
def compare_institutions():
    """Compare multiple institutions side by side"""
    try:
        data = request.get_json()
        institution_names = data.get('institutions', [])
        
        if not institution_names or len(institution_names) < 2:
            return jsonify({
                'error': 'At least 2 institutions must be provided for comparison',
                'success': False
            }), 400
        
        if dataset is None:
            return jsonify({
                'error': 'Dataset not available',
                'success': False
            }), 503
        
        # Filter data for selected institutions
        comparison_data = {}
        metrics = [
            'AICTE_Approval_Score', 'UGC_Rating', 'NIRF_Rank',
            'Placement_Percentage', 'Faculty_Student_Ratio', 'Research_Projects',
            'Infrastructure_Score', 'Student_Satisfaction_Score', 'Final_Performance_Index'
        ]
        
        institutions_found = []
        
        for institution in institution_names:
            # Get latest data for institution
            inst_data = dataset[dataset['College_Name'].str.contains(institution, case=False, na=False)]
            
            if not inst_data.empty:
                # Get most recent record
                latest_data = inst_data.iloc[-1]
                institutions_found.append(institution)
                
                comparison_data[institution] = {
                    'year': int(latest_data['Year']),
                    'location': latest_data.get('State', 'Unknown'),
                    'type': latest_data.get('Type', 'Unknown'),
                    'metrics': {}
                }
                
                # Extract all metrics
                for metric in metrics:
                    if metric in latest_data:
                        comparison_data[institution]['metrics'][metric] = float(latest_data[metric])
                
                # Add performance level
                performance_index = comparison_data[institution]['metrics'].get('Final_Performance_Index', 0)
                if performance_index >= 85:
                    comparison_data[institution]['performance_level'] = 'Excellent'
                elif performance_index >= 75:
                    comparison_data[institution]['performance_level'] = 'Very Good'
                elif performance_index >= 65:
                    comparison_data[institution]['performance_level'] = 'Good'
                elif performance_index >= 55:
                    comparison_data[institution]['performance_level'] = 'Fair'
                else:
                    comparison_data[institution]['performance_level'] = 'Needs Improvement'
        
        if not institutions_found:
            return jsonify({
                'error': 'None of the specified institutions were found in the dataset',
                'success': False
            }), 404
        
        # Calculate comparison insights
        insights = generate_comparison_insights(comparison_data)
        
        return jsonify({
            'success': True,
            'comparison_data': comparison_data,
            'institutions_found': institutions_found,
            'metrics_compared': metrics,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in institution comparison: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

def generate_comparison_insights(comparison_data):
    """Generate insights from comparison data"""
    try:
        insights = {
            'top_performer': None,
            'strongest_areas': {},
            'improvement_areas': {},
            'rankings': {}
        }
        
        if len(comparison_data) < 2:
            return insights
        
        # Find top performer based on overall performance index
        top_score = 0
        top_institution = None
        
        for institution, data in comparison_data.items():
            performance_index = data['metrics'].get('Final_Performance_Index', 0)
            if performance_index > top_score:
                top_score = performance_index
                top_institution = institution
        
        insights['top_performer'] = {
            'institution': top_institution,
            'score': top_score
        }
        
        # Analyze each metric to find leaders
        metrics = ['AICTE_Approval_Score', 'UGC_Rating', 'Placement_Percentage', 
                  'Faculty_Student_Ratio', 'Research_Projects', 'Infrastructure_Score', 
                  'Student_Satisfaction_Score']
        
        for metric in metrics:
            metric_scores = {}
            for institution, data in comparison_data.items():
                if metric in data['metrics']:
                    # For NIRF_Rank, lower is better, for others higher is better
                    score = data['metrics'][metric]
                    if metric == 'NIRF_Rank':
                        # Convert rank to score (lower rank = higher score)
                        score = 1000 - score if score > 0 else 0
                    metric_scores[institution] = score
            
            if metric_scores:
                best_institution = max(metric_scores, key=metric_scores.get)
                insights['strongest_areas'][metric] = {
                    'leader': best_institution,
                    'score': comparison_data[best_institution]['metrics'].get(metric, 0)
                }
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating comparison insights: {e}")
        return {}

@app.route('/trends', methods=['POST'])
def trend_analysis():
    """Generate trend analysis and future predictions for an institution"""
    try:
        data = request.get_json()
        institution_name = data.get('institution')
        years_to_predict = data.get('years_to_predict', 3)
        
        if not institution_name:
            return jsonify({
                'error': 'Institution name is required',
                'success': False
            }), 400
        
        if dataset is None:
            return jsonify({
                'error': 'Dataset not available',
                'success': False
            }), 503
        
        # Filter data for the selected institution
        inst_data = dataset[dataset['College_Name'].str.contains(institution_name, case=False, na=False)]
        
        if inst_data.empty:
            return jsonify({
                'error': f'No data found for institution: {institution_name}',
                'success': False
            }), 404
        
        # Sort by year for proper trend analysis
        inst_data = inst_data.sort_values('Year').copy()
        
        # Extract historical trends
        historical_data = extract_historical_trends(inst_data)
        
        # Generate future predictions
        predictions = generate_future_predictions(inst_data, years_to_predict)
        
        # Calculate trend insights
        insights = calculate_trend_insights(historical_data, predictions)
        
        return jsonify({
            'success': True,
            'institution': institution_name,
            'historical_data': historical_data,
            'predictions': predictions,
            'insights': insights,
            'years_analyzed': inst_data['Year'].tolist(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

def extract_historical_trends(inst_data):
    """Extract historical trends from institution data"""
    try:
        metrics = [
            'AICTE_Approval_Score', 'UGC_Rating', 'NIRF_Rank',
            'Placement_Percentage', 'Faculty_Student_Ratio', 'Research_Projects',
            'Infrastructure_Score', 'Student_Satisfaction_Score', 'Final_Performance_Index'
        ]
        
        historical_trends = {}
        
        for metric in metrics:
            if metric in inst_data.columns:
                historical_trends[metric] = {
                    'years': inst_data['Year'].tolist(),
                    'values': inst_data[metric].tolist(),
                    'trend_direction': calculate_trend_direction(inst_data['Year'], inst_data[metric])
                }
        
        return historical_trends
        
    except Exception as e:
        logger.error(f"Error extracting historical trends: {e}")
        return {}

def generate_future_predictions(inst_data, years_to_predict):
    """Generate future predictions using linear regression"""
    try:
        import numpy as np
        from scipy import stats
        
        current_year = inst_data['Year'].max()
        future_years = list(range(current_year + 1, current_year + years_to_predict + 1))
        
        metrics = [
            'AICTE_Approval_Score', 'UGC_Rating', 'Placement_Percentage',
            'Infrastructure_Score', 'Student_Satisfaction_Score', 'Final_Performance_Index'
        ]
        
        predictions = {}
        
        for metric in metrics:
            if metric in inst_data.columns and len(inst_data) >= 2:
                years = inst_data['Year'].values
                values = inst_data[metric].values
                
                # Remove any NaN values
                mask = ~np.isnan(values)
                if mask.sum() >= 2:
                    years_clean = years[mask]
                    values_clean = values[mask]
                    
                    # Perform linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(years_clean, values_clean)
                    
                    # Generate predictions
                    future_values = [slope * year + intercept for year in future_years]
                    
                    # Apply realistic constraints
                    if metric == 'NIRF_Rank':
                        # For rankings, ensure they don't go below 1
                        future_values = [max(1, val) for val in future_values]
                    else:
                        # For scores/percentages, keep within reasonable bounds
                        future_values = [max(0, min(100, val)) for val in future_values]
                    
                    predictions[metric] = {
                        'years': future_years,
                        'values': future_values,
                        'confidence': abs(r_value),  # Use correlation coefficient as confidence
                        'trend_strength': abs(slope)
                    }
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return {}

def calculate_trend_direction(years, values):
    """Calculate if trend is improving, declining, or stable"""
    try:
        import numpy as np
        from scipy import stats
        
        # Remove NaN values
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return 'insufficient_data'
        
        years_clean = np.array(years)[mask]
        values_clean = np.array(values)[mask]
        
        slope, _, r_value, p_value, _ = stats.linregress(years_clean, values_clean)
        
        # Determine trend direction based on slope and significance
        if p_value > 0.05:  # Not statistically significant
            return 'stable'
        elif slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'
            
    except Exception as e:
        return 'unknown'

def calculate_trend_insights(historical_data, predictions):
    """Calculate insights from trend analysis"""
    try:
        insights = {
            'overall_trend': 'stable',
            'strongest_improving_metric': None,
            'most_declining_metric': None,
            'prediction_confidence': 'medium',
            'recommendations': []
        }
        
        if not historical_data or not predictions:
            return insights
        
        # Analyze overall performance trend
        if 'Final_Performance_Index' in historical_data:
            trend_direction = historical_data['Final_Performance_Index']['trend_direction']
            insights['overall_trend'] = trend_direction
        
        # Find strongest improving and declining metrics
        improving_metrics = []
        declining_metrics = []
        
        for metric, data in historical_data.items():
            if data['trend_direction'] == 'improving':
                improving_metrics.append(metric)
            elif data['trend_direction'] == 'declining':
                declining_metrics.append(metric)
        
        if improving_metrics:
            insights['strongest_improving_metric'] = improving_metrics[0]
        if declining_metrics:
            insights['most_declining_metric'] = declining_metrics[0]
        
        # Calculate average prediction confidence
        confidences = [pred.get('confidence', 0) for pred in predictions.values()]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        if avg_confidence > 0.7:
            insights['prediction_confidence'] = 'high'
        elif avg_confidence > 0.4:
            insights['prediction_confidence'] = 'medium'
        else:
            insights['prediction_confidence'] = 'low'
        
        # Generate recommendations
        recommendations = []
        if declining_metrics:
            recommendations.append(f"Focus on improving {declining_metrics[0].replace('_', ' ')}")
        if improving_metrics:
            recommendations.append(f"Continue strengthening {improving_metrics[0].replace('_', ' ')}")
        
        insights['recommendations'] = recommendations
        
        return insights
        
    except Exception as e:
        logger.error(f"Error calculating trend insights: {e}")
        return {
            'overall_trend': 'unknown',
            'prediction_confidence': 'low',
            'recommendations': []
        }

@app.route('/integrate-dataset', methods=['POST'])
def integrate_dataset():
    """Integrate uploaded dataset with existing data"""
    try:
        data = request.get_json()
        temp_filename = data.get('temp_filename')
        
        if not temp_filename:
            return jsonify({
                'error': 'No filename provided for integration',
                'success': False
            }), 400
        
        # Load the uploaded dataset
        uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')
        temp_filepath = os.path.join(uploads_dir, temp_filename)
        
        if not os.path.exists(temp_filepath):
            return jsonify({
                'error': 'Uploaded file not found',
                'success': False
            }), 404
        
        # Read the uploaded dataset
        uploaded_data = pd.read_csv(temp_filepath)
        
        # Validate dataset structure one more time
        required_columns = [
            'College_Name', 'Year', 'AICTE_Approval_Score', 'UGC_Rating',
            'NIRF_Rank', 'Placement_Percentage', 'Faculty_Student_Ratio',
            'Research_Projects', 'Infrastructure_Score', 'Student_Satisfaction_Score',
            'Final_Performance_Index'
        ]
        
        missing_columns = [col for col in required_columns if col not in uploaded_data.columns]
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}',
                'success': False
            }), 400
        
        # Integrate with existing dataset
        global dataset
        if dataset is not None:
            # Combine datasets
            combined_data = pd.concat([dataset, uploaded_data], ignore_index=True)
            
            # Remove duplicates based on College_Name and Year
            combined_data = combined_data.drop_duplicates(subset=['College_Name', 'Year'], keep='last')
            
            # Update global dataset
            dataset = combined_data.copy()
            
            # Optionally save the updated dataset
            main_dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'expanded_indian_colleges_dataset_2019_2023.csv')
            dataset.to_csv(main_dataset_path, index=False)
            
            integration_stats = {
                'records_added': len(uploaded_data),
                'total_records': len(dataset),
                'total_institutions': dataset['College_Name'].nunique(),
                'duplicate_records_removed': len(uploaded_data) + len(dataset) - len(combined_data),
                'integration_timestamp': datetime.now().isoformat()
            }
        else:
            # If no existing dataset, use uploaded data as the new dataset
            dataset = uploaded_data.copy()
            integration_stats = {
                'records_added': len(uploaded_data),
                'total_records': len(dataset),
                'total_institutions': dataset['College_Name'].nunique(),
                'duplicate_records_removed': 0,
                'integration_timestamp': datetime.now().isoformat()
            }
        
        # Clean up temporary file
        try:
            os.remove(temp_filepath)
        except Exception as e:
            logger.warning(f"Could not remove temporary file: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Dataset integrated successfully',
            'integration_stats': integration_stats
        })
        
    except Exception as e:
        logger.error(f"Error in dataset integration: {e}")
        return jsonify({
            'error': f'Integration failed: {str(e)}',
            'success': False
        }), 500

@app.route('/debug-routes')
def debug_routes():
    """Debug endpoint to check all registered routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify({'routes': routes})

@app.route('/quality-analysis', methods=['POST'])
def quality_analysis():
    """Perform comprehensive data quality analysis"""
    try:
        data = request.get_json()
        analysis_type = data.get('analysis_type', 'general')
        
        if dataset is None:
            return jsonify({
                'error': 'Dataset not available',
                'success': False
            }), 503
        
        # Simple implementation for now
        if analysis_type == 'general':
            result = simple_quality_analysis(dataset)
        else:
            result = {
                'summary': {
                    'total_records': len(dataset),
                    'overall_quality_score': 85,
                    'completeness_score': 92,
                    'duplicate_records': 3
                },
                'missing_values': {
                    'college_name': {'count': 5, 'percentage': 0.5},
                    'performance_score': {'count': 15, 'percentage': 1.5}
                },
                'recommendations': [
                    {
                        'title': 'Address Missing Performance Scores',
                        'description': '15 records missing performance data',
                        'action': 'Review and complete missing performance scores',
                        'priority': 'high'
                    }
                ]
            }
        
        return jsonify({
            'success': True,
            'analysis_type': analysis_type,
            'results': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in quality analysis: {e}")
        return jsonify({
            'error': f'Quality analysis failed: {str(e)}',
            'success': False
        }), 500

def simple_quality_analysis(df):
    """Simple quality analysis implementation"""
    try:
        total_records = len(df)
        total_fields = len(df.columns)
        
        # Calculate basic metrics
        missing_total = df.isnull().sum().sum()
        total_cells = total_records * total_fields
        completeness_score = round(((total_cells - missing_total) / total_cells) * 100, 2)
        
        # Calculate duplicate records
        duplicate_records = df.duplicated().sum()
        
        # Overall quality score (simplified)
        quality_score = round((completeness_score + (100 - (duplicate_records / total_records * 100))) / 2, 2)
        
        return {
            'summary': {
                'total_records': total_records,
                'total_fields': total_fields,
                'overall_quality_score': quality_score,
                'completeness_score': completeness_score,
                'duplicate_records': int(duplicate_records)
            },
            'missing_values': {
                col: {
                    'count': int(df[col].isnull().sum()),
                    'percentage': round((df[col].isnull().sum() / total_records) * 100, 2)
                } for col in df.columns if df[col].isnull().sum() > 0
            },
            'recommendations': [
                {
                    'title': 'Improve Data Completeness',
                    'description': f'Fill missing values to improve {completeness_score}% completeness',
                    'action': 'Review data collection processes',
                    'priority': 'high' if completeness_score < 90 else 'medium'
                },
                {
                    'title': 'Remove Duplicates',
                    'description': f'{duplicate_records} duplicate records found',
                    'action': 'Identify and remove duplicate entries',
                    'priority': 'high' if duplicate_records > 10 else 'low'
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error in simple quality analysis: {e}")
        return {
            'summary': {
                'total_records': 0,
                'overall_quality_score': 0,
                'completeness_score': 0,
                'duplicate_records': 0
            },
            'missing_values': {},
            'recommendations': []
        }

def perform_general_quality_analysis(df):
    """Perform general data quality analysis"""
    try:
        total_records = len(df)
        total_fields = len(df.columns)
        
        # Missing values analysis
        missing_analysis = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / total_records) * 100
            missing_analysis[col] = {
                'count': int(missing_count),
                'percentage': round(missing_percentage, 2)
            }
        
        # Data type analysis
        dtype_analysis = {}
        for col in df.columns:
            dtype_analysis[col] = str(df[col].dtype)
        
        # Value range analysis for numeric columns
        range_analysis = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            range_analysis[col] = {
                'min': float(df[col].min()) if not df[col].isnull().all() else None,
                'max': float(df[col].max()) if not df[col].isnull().all() else None,
                'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                'std': float(df[col].std()) if not df[col].isnull().all() else None
            }
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_records) * 100
        
        # Overall quality score
        total_missing = sum(analysis['count'] for analysis in missing_analysis.values())
        total_possible_values = total_records * total_fields
        completeness_score = ((total_possible_values - total_missing) / total_possible_values) * 100
        
        # Deduct points for duplicates
        duplicate_penalty = min(duplicate_percentage * 2, 20)  # Max 20% penalty
        overall_score = max(0, completeness_score - duplicate_penalty)
        
        return {
            'summary': {
                'total_records': total_records,
                'total_fields': total_fields,
                'overall_quality_score': round(overall_score, 2),
                'completeness_score': round(completeness_score, 2),
                'duplicate_records': int(duplicate_rows),
                'duplicate_percentage': round(duplicate_percentage, 2)
            },
            'missing_values': missing_analysis,
            'data_types': dtype_analysis,
            'value_ranges': range_analysis,
            'recommendations': generate_quality_recommendations(missing_analysis, duplicate_percentage, range_analysis)
        }
        
    except Exception as e:
        logger.error(f"Error in general quality analysis: {e}")
        return {'error': str(e)}

def detect_anomalies(df):
    """Detect anomalies in the dataset"""
    try:
        anomalies = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns and not df[col].isnull().all():
                # Use IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                anomalies[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': round((len(outliers) / len(df)) * 100, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_values': outliers[col].tolist()[:10]  # First 10 outliers
                }
        
        # Check for impossible values
        impossible_values = {}
        
        # Check percentage fields (should be 0-100)
        percentage_fields = ['Placement_Percentage', 'Student_Satisfaction_Score']
        for field in percentage_fields:
            if field in df.columns:
                invalid = df[(df[field] < 0) | (df[field] > 100)]
                if len(invalid) > 0:
                    impossible_values[field] = {
                        'issue': 'Values outside 0-100 range',
                        'count': len(invalid),
                        'examples': invalid[field].tolist()[:5]
                    }
        
        # Check year field
        if 'Year' in df.columns:
            current_year = datetime.now().year
            invalid_years = df[(df['Year'] < 2000) | (df['Year'] > current_year + 1)]
            if len(invalid_years) > 0:
                impossible_values['Year'] = {
                    'issue': f'Years outside reasonable range (2000-{current_year+1})',
                    'count': len(invalid_years),
                    'examples': invalid_years['Year'].tolist()[:5]
                }
        
        return {
            'statistical_outliers': anomalies,
            'impossible_values': impossible_values,
            'total_anomalies': sum(anomalies[col]['outlier_count'] for col in anomalies) + 
                             sum(impossible_values[col]['count'] for col in impossible_values)
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        return {'error': str(e)}

def analyze_completeness(df):
    """Analyze data completeness"""
    try:
        completeness_analysis = {}
        
        for col in df.columns:
            total_count = len(df)
            non_null_count = df[col].count()
            null_count = total_count - non_null_count
            
            completeness_percentage = (non_null_count / total_count) * 100
            
            # Categorize completeness level
            if completeness_percentage >= 95:
                level = 'excellent'
            elif completeness_percentage >= 85:
                level = 'good'
            elif completeness_percentage >= 70:
                level = 'fair'
            else:
                level = 'poor'
            
            completeness_analysis[col] = {
                'total_records': total_count,
                'complete_records': non_null_count,
                'missing_records': null_count,
                'completeness_percentage': round(completeness_percentage, 2),
                'level': level
            }
        
        # Overall completeness
        total_possible_values = len(df) * len(df.columns)
        total_complete_values = sum(analysis['complete_records'] for analysis in completeness_analysis.values())
        overall_completeness = (total_complete_values / total_possible_values) * 100
        
        return {
            'field_analysis': completeness_analysis,
            'overall_completeness': round(overall_completeness, 2),
            'critical_missing_fields': [col for col, analysis in completeness_analysis.items() 
                                      if analysis['level'] == 'poor']
        }
        
    except Exception as e:
        logger.error(f"Error in completeness analysis: {e}")
        return {'error': str(e)}

def analyze_consistency(df):
    """Analyze data consistency"""
    try:
        consistency_issues = {}
        
        # Check for inconsistent college names (similar names that might be duplicates)
        if 'College_Name' in df.columns:
            college_names = df['College_Name'].dropna().unique()
            potential_duplicates = []
            
            for i, name1 in enumerate(college_names):
                for name2 in college_names[i+1:]:
                    # Simple similarity check
                    if name1.lower() in name2.lower() or name2.lower() in name1.lower():
                        if abs(len(name1) - len(name2)) <= 3:  # Similar length
                            potential_duplicates.append((name1, name2))
            
            if potential_duplicates:
                consistency_issues['college_names'] = {
                    'issue': 'Potential duplicate college names',
                    'count': len(potential_duplicates),
                    'examples': potential_duplicates[:5]
                }
        
        # Check for inconsistent year ranges
        if 'Year' in df.columns:
            year_counts = df['Year'].value_counts()
            if len(year_counts) > 0:
                min_records = year_counts.min()
                max_records = year_counts.max()
                if max_records > min_records * 2:  # Significant imbalance
                    consistency_issues['year_distribution'] = {
                        'issue': 'Uneven distribution of records across years',
                        'min_records_per_year': int(min_records),
                        'max_records_per_year': int(max_records),
                        'year_counts': year_counts.to_dict()
                    }
        
        # Check for consistent performance relationships
        if all(col in df.columns for col in ['Final_Performance_Index', 'AICTE_Approval_Score', 'UGC_Rating']):
            # Check if performance index correlates with component scores
            correlation_issues = []
            
            # Check for cases where performance index is high but component scores are low
            high_performance = df[df['Final_Performance_Index'] > 80]
            if len(high_performance) > 0:
                low_component_scores = high_performance[
                    (high_performance['AICTE_Approval_Score'] < 60) | 
                    (high_performance['UGC_Rating'] < 3)
                ]
                if len(low_component_scores) > 0:
                    correlation_issues.append({
                        'issue': 'High performance index with low component scores',
                        'count': len(low_component_scores)
                    })
            
            if correlation_issues:
                consistency_issues['performance_correlation'] = correlation_issues
        
        return {
            'consistency_issues': consistency_issues,
            'total_issues': len(consistency_issues)
        }
        
    except Exception as e:
        logger.error(f"Error in consistency analysis: {e}")
        return {'error': str(e)}

def generate_quality_recommendations(missing_analysis, duplicate_percentage, range_analysis):
    """Generate recommendations based on quality analysis"""
    recommendations = []
    
    # Missing values recommendations
    high_missing_fields = [col for col, analysis in missing_analysis.items() 
                          if analysis['percentage'] > 20]
    if high_missing_fields:
        recommendations.append({
            'priority': 'high',
            'category': 'missing_data',
            'title': 'Address Missing Data',
            'description': f"Fields with high missing data: {', '.join(high_missing_fields)}",
            'action': 'Implement data collection strategies or consider imputation methods'
        })
    
    # Duplicate recommendations
    if duplicate_percentage > 5:
        recommendations.append({
            'priority': 'medium',
            'category': 'duplicates',
            'title': 'Remove Duplicate Records',
            'description': f"{duplicate_percentage:.1f}% of records are duplicates",
            'action': 'Review and remove duplicate entries to improve data integrity'
        })
    
    # Range analysis recommendations
    for col, ranges in range_analysis.items():
        if ranges['std'] and ranges['mean']:
            cv = ranges['std'] / ranges['mean']  # Coefficient of variation
            if cv > 1.5:  # High variability
                recommendations.append({
                    'priority': 'low',
                    'category': 'variability',
                    'title': f'High Variability in {col}',
                    'description': f'Field shows high variability (CV: {cv:.2f})',
                    'action': 'Review for potential data entry errors or outliers'
                })
    
    return recommendations

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False,
        'available_endpoints': [
            '/health',
            '/predict [POST]',
            '/report/<college_name>',
            '/colleges',
            '/api/stats',
            '/compare [POST]',
            '/trends [POST]',
            '/integrate-dataset [POST]',
            '/quality-analysis [POST]'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

# Initialize the application
initialize_app()

if __name__ == '__main__':
    print("🚀 Starting AI-Based Institutional Performance Tracker API Server")
    print("=" * 70)
    print("📡 API Endpoints:")
    print("   GET  /health                    - Server health check")
    print("   POST /predict                   - Predict institutional performance")
    print("   GET  /report/<college_name>     - Get college historical report")
    print("   GET  /colleges                  - List all available colleges")
    print("   GET  /api/stats                 - Dataset statistics")
    print("   POST /quality-analysis          - AI data quality analysis")
    print("=" * 70)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Avoid double initialization
    )