"""
APP_ENHANCED.PY - Smart Farming Recommendation System
Flask Web Application with 4 ML Models (Decision Tree, Random Forest, KNN, SVM)

Features:
- Loads all 4 trained ML models
- User can select which model to use
- Makes predictions with selected model
- Displays results with model comparison
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import requests
from waitress import serve
from functools import lru_cache
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
CORS(app)

# Global variables for models
models = {
    'dt': None,           # Decision Tree
    'rf': None,           # Random Forest
    'knn': None,          # K-Nearest Neighbors
    'svm': None,          # Support Vector Machine
    'ensemble': None
}
scaler = None
feature_names = None
model_info = {
    'dt': {
        'name': 'Decision Tree',
        'description': 'Fast, interpretable tree-based classifier',
        'pros': ['Fast training', 'Interpretable', 'Good baseline'],
        'cons': ['Prone to overfitting', 'Can be unstable'],
        'best_for': 'Quick predictions, explanations'
    },
    'rf': {
        'name': 'Random Forest',
        'description': 'Ensemble of decision trees',
        'pros': ['High accuracy', 'Reduces overfitting', 'Feature importance'],
        'cons': ['Slower than single tree', 'Less interpretable'],
        'best_for': 'Balanced accuracy and speed'
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'description': 'Distance-based instance learning',
        'pros': ['Simple', 'Good for non-linear', 'No training needed'],
        'cons': ['Slow predictions', 'Memory intensive', 'Sensitive to scaling'],
        'best_for': 'Non-linear patterns'
    },
    'svm': {
        'name': 'Support Vector Machine',
        'description': 'Margin-based classifier',
        'pros': ['Good generalization', 'Handles high dimensions'],
        'cons': ['Slower training', 'Less interpretable'],
        'best_for': 'Complex boundaries'
    },
    'ensemble': {
        'name': 'Ensemble (Majority Vote)',
        'description': 'Combines all 4 models for maximum accuracy',
        'pros': ['Highest overall accuracy', 'Robust to outliers'],
        'cons': ['Slower prediction time'],
        'best_for': 'Maximum confidence predictions'
    }
}

crop_info = {
    'rice': {
        'crop': 'Rice',
        'best_season': 'Kharif',
        'soil_type': 'Clay loam',
        'water_requirement': 'High',
        'spacing': '20 x 15 cm',
        'notes': 'Requires standing water and a warm, humid climate.'
    },
    'maize': {
        'crop': 'Maize',
        'best_season': 'Kharif/Rabi',
        'soil_type': 'Loamy',
        'water_requirement': 'Moderate',
        'spacing': '45 x 25 cm',
        'notes': 'Adaptable crop with good yield in warm conditions.'
    },
    'cotton': {
        'crop': 'Cotton',
        'best_season': 'Kharif',
        'soil_type': 'Black/Loamy',
        'water_requirement': 'Moderate',
        'spacing': '75 x 45 cm',
        'notes': 'Requires warm temperatures and well-drained soil.'
    },
    'papaya': {
        'crop': 'Papaya',
        'best_season': 'Tropical',
        'soil_type': 'Loamy',
        'water_requirement': 'High',
        'spacing': '3 x 3 m',
        'notes': 'Fast-growing fruit crop with high water needs.'
    },
    'coconut': {
        'crop': 'Coconut',
        'best_season': 'Tropical',
        'soil_type': 'Sandy loam',
        'water_requirement': 'High',
        'spacing': '7.5 x 7.5 m',
        'notes': 'Best grown in coastal or humid tropical regions.'
    },
    'orange': {
        'crop': 'Orange',
        'best_season': 'Winter',
        'soil_type': 'Loamy',
        'water_requirement': 'Moderate',
        'spacing': '6 x 6 m',
        'notes': 'Requires well-drained soil and moderate water.'
    },
    'apple': {
        'crop': 'Apple',
        'best_season': 'Temperate',
        'soil_type': 'Loamy',
        'water_requirement': 'Moderate',
        'spacing': '5 x 5 m',
        'notes': 'Needs cool winters and frost-free blossom period.'
    },
    'banana': {
        'crop': 'Banana',
        'best_season': 'Tropical',
        'soil_type': 'Loamy',
        'water_requirement': 'High',
        'spacing': '2.5 x 2.5 m',
        'notes': 'Requires rich soil and consistent moisture.'
    },
    'blackgram': {
        'crop': 'Black Gram',
        'best_season': 'Kharif',
        'soil_type': 'Sandy loam',
        'water_requirement': 'Low',
        'spacing': '30 x 10 cm',
        'notes': 'Drought-tolerant pulse crop for light soils.'
    },
    'chickpea': {
        'crop': 'Chickpea',
        'best_season': 'Rabi',
        'soil_type': 'Loam',
        'water_requirement': 'Low',
        'spacing': '30 x 10 cm',
        'notes': 'Requires cool climate and well-drained soil.'
    },
    'coffee': {
        'crop': 'Coffee',
        'best_season': 'Tropical',
        'soil_type': 'Acidic loam',
        'water_requirement': 'Moderate',
        'spacing': '2.5 x 2.5 m',
        'notes': 'Prefers shaded conditions and regular rainfall.'
    },
    'grapes': {
        'crop': 'Grapes',
        'best_season': 'Temperate',
        'soil_type': 'Loamy',
        'water_requirement': 'Moderate',
        'spacing': '2 x 1.5 m',
        'notes': 'Needs trellis support and good sun exposure.'
    },
    'jute': {
        'crop': 'Jute',
        'best_season': 'Kharif',
        'soil_type': 'Clay loam',
        'water_requirement': 'High',
        'spacing': '30 x 10 cm',
        'notes': 'Grows best in warm, humid environments with plenty of water.'
    },
    'kidneybeans': {
        'crop': 'Kidney Beans',
        'best_season': 'Kharif/Rabi',
        'soil_type': 'Loamy',
        'water_requirement': 'Moderate',
        'spacing': '30 x 15 cm',
        'notes': 'Requires consistent moisture and sunlight.'
    },
    'lentil': {
        'crop': 'Lentil',
        'best_season': 'Rabi',
        'soil_type': 'Loamy',
        'water_requirement': 'Low',
        'spacing': '25 x 10 cm',
        'notes': 'Cool-season pulse crop with low water needs.'
    },
    'mango': {
        'crop': 'Mango',
        'best_season': 'Summer',
        'soil_type': 'Loamy',
        'water_requirement': 'Moderate',
        'spacing': '10 x 10 m',
        'notes': 'Requires warm, dry weather for flowering and fruit set.'
    },
    'mothbeans': {
        'crop': 'Moth Beans',
        'best_season': 'Kharif',
        'soil_type': 'Sandy',
        'water_requirement': 'Low',
        'spacing': '30 x 10 cm',
        'notes': 'Very drought-tolerant pulse crop for arid regions.'
    },
    'mungbean': {
        'crop': 'Mung Bean',
        'best_season': 'Kharif',
        'soil_type': 'Loamy',
        'water_requirement': 'Low',
        'spacing': '30 x 10 cm',
        'notes': 'Fast-growing pulse crop well suited for warm conditions.'
    },
    'muskmelon': {
        'crop': 'Muskmelon',
        'best_season': 'Summer',
        'soil_type': 'Sandy loam',
        'water_requirement': 'Moderate',
        'spacing': '120 x 60 cm',
        'notes': 'Requires warm temperatures and good drainage.'
    },
    'orange': {
        'crop': 'Orange',
        'best_season': 'Winter',
        'soil_type': 'Loamy',
        'water_requirement': 'Moderate',
        'spacing': '6 x 6 m',
        'notes': 'Needs adequate sunlight and well-drained soil.'
    },
    'pomegranate': {
        'crop': 'Pomegranate',
        'best_season': 'Summer',
        'soil_type': 'Loamy',
        'water_requirement': 'Low',
        'spacing': '4 x 4 m',
        'notes': 'Drought-tolerant fruit crop with low water needs.'
    },
    'pigeonpeas': {
        'crop': 'Pigeon Peas',
        'best_season': 'Kharif',
        'soil_type': 'Loamy',
        'water_requirement': 'Moderate',
        'spacing': '45 x 15 cm',
        'notes': 'Legume crop with deep roots for dryland farming.'
    },
    'watermelon': {
        'crop': 'Watermelon',
        'best_season': 'Summer',
        'soil_type': 'Sandy loam',
        'water_requirement': 'High',
        'spacing': '180 x 90 cm',
        'notes': 'Needs warm weather and plenty of water for sweet fruits.'
    }
}

def get_crop_details(crop_name):
    if not crop_name:
        return None
    return crop_info.get(str(crop_name).strip().lower(), {
        'crop': str(crop_name),
        'best_season': 'Unknown',
        'soil_type': 'Unknown',
        'water_requirement': 'Unknown',
        'spacing': 'Unknown',
        'notes': 'No specific crop details available.'
    })

# ============================================================================
# LOAD MODELS AT STARTUP
# ============================================================================

crop_feature_ranges = {}

def calculate_crop_ranges():
    global crop_feature_ranges
    try:
        if os.path.exists('crop_data.csv'):
            df = pd.read_csv('crop_data.csv')
            for crop_name in df['label'].unique():
                crop_df = df[df['label'] == crop_name]
                crop_feature_ranges[crop_name] = {}
                for feature in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
                    crop_feature_ranges[crop_name][feature] = {
                        'min': float(crop_df[feature].min()),
                        'max': float(crop_df[feature].max()),
                        'mean': float(crop_df[feature].mean())
                    }
            print("✓ Crop agronomical ranges precalculated from dataset")
    except Exception as e:
        print(f"✗ Error calculating crop ranges: {e}")


def load_models():
    """Load all trained models and scaler from disk"""
    global models, scaler, feature_names
    
    try:
        print("Loading trained models...")
        
        # Load Decision Tree
        with open('models/crop_recommendation_model.pkl', 'rb') as f:
            models['dt'] = pickle.load(f)
        print("✓ Decision Tree model loaded")
        
        # Load Random Forest
        with open('models/random_forest_model.pkl', 'rb') as f:
            models['rf'] = pickle.load(f)
        print("✓ Random Forest model loaded")
        
        # Load KNN
        with open('models/knn_model.pkl', 'rb') as f:
            models['knn'] = pickle.load(f)
        print("✓ KNN model loaded")
        
        # Load SVM
        with open('models/svm_model.pkl', 'rb') as f:
            models['svm'] = pickle.load(f)
        print("✓ SVM model loaded")
        
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded")
        
        # Define feature names
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Precalculate crop ranges
        calculate_crop_ranges()
        
        print("✓ All models loaded successfully!\n")
        return True
    
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("  Make sure to run: python train_model_enhanced.py")
        return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_feature_importance(model, feature_names, top_n=7):
    """Get feature importance from tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_scores = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(top_n)
    return {
        'features': top_features['feature'].tolist(),
        'scores': [float(score) for score in top_features['importance'].tolist()]
    }


def get_alternative_predictions(model, features, scaler, top_k=3):
    """Get top K predictions from model"""
    features_scaled = scaler.transform([features])
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        classes = model.classes_
        
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        predictions = [
            {
                'crop': str(classes[idx]),
                'confidence': float(probabilities[idx]) * 100
            }
            for idx in top_indices
        ]
    else:
        prediction = model.predict(features_scaled)[0]
        predictions = [{'crop': str(prediction), 'confidence': 100.0}]
    
    return predictions


def validate_input(data):
    """Validate user input"""
    try:
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing field: {field}"
        
        ranges = {
            'N': (0, 140),
            'P': (0, 150),
            'K': (0, 210),
            'temperature': (-10, 50),
            'humidity': (0, 100),
            'ph': (0, 14),
            'rainfall': (0, 300)
        }
        
        for field, (min_val, max_val) in ranges.items():
            value = float(data[field])
            if not (min_val <= value <= max_val):
                return False, f"{field} out of range: {min_val}-{max_val}"
        
        return True, ""
    
    except Exception as e:
        return False, f"Invalid input: {str(e)}"


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the home page"""
    return render_template('index.html')


@app.route('/recommend')
def recommend():
    """Serve the recommendation page"""
    return render_template('recommend.html')


@app.route('/about')
def about():
    """Serve the about page"""
    return render_template('about.html')


@app.route('/api/crops', methods=['GET'])
def get_crops():
    """Get agronomical details of all supported crops"""
    try:
        sorted_crops = sorted(crop_info.values(), key=lambda x: x['crop'])
        return jsonify({
            'success': True,
            'crops': sorted_crops
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@lru_cache(maxsize=128)
def fetch_geocoding_data(place_name):
    # Try Nominatim first
    nominatim_params = {
        'q': place_name,
        'format': 'json',
        'limit': 1,
        'addressdetails': 1
    }
    headers = {'User-Agent': 'SmartFarmingCropAdvisorySystem/1.0 (jairamkurapati@gmail.com)'}
    
    response = requests.get("https://nominatim.openstreetmap.org/search", params=nominatim_params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
        
    if data:
        result = data[0]
        address = result.get('address', {})
        location_name = result.get('display_name', place_name).split(',')[0]
        state = address.get('state', address.get('region', ''))
        country = address.get('country', '')
        
        formatted_name = f"{location_name}"
        if state:
            formatted_name += f", {state}"
        if country:
            formatted_name += f" ({country})"
            
        return {
            'success': True,
            'latitude': float(result['lat']),
            'longitude': float(result['lon']),
            'name': formatted_name
        }
        
    # Fallback to ArcGIS if Nominatim fails (very good for small villages in India)
    arcgis_params = {
        'f': 'json',
        'singleLine': f"{place_name}, India",
        'outFields': 'Match_addr,Addr_type'
    }
    arcgis_response = requests.get("https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates", params=arcgis_params, timeout=10)
    arcgis_response.raise_for_status()
    arcgis_data = arcgis_response.json()
        
    candidates = arcgis_data.get('candidates', [])
    if not candidates:
        return {'success': False, 'error': f"Location '{place_name}' not found.", 'status_code': 404}
        
    best_match = candidates[0]
    formatted_name = best_match.get('address', place_name)
    location = best_match.get('location', {})
    
    return {
        'success': True,
        'latitude': float(location.get('y', 0)),
        'longitude': float(location.get('x', 0)),
        'name': formatted_name
    }

@app.route('/api/geocode', methods=['GET'])
def geocode_city():
    """Geocode a place name (city, town, or village) using OpenStreetMap Nominatim with ArcGIS fallback"""
    place_name = request.args.get('name', '').strip()
    if not place_name:
        return jsonify({'error': 'Name parameter is required'}), 400
        
    try:
        res = fetch_geocoding_data(place_name)
        if not res.get('success', False) and 'status_code' in res:
            return jsonify(res), res['status_code']
        return jsonify(res), 200
    except Exception as e:
        return jsonify({'error': f"Geocoding error: {str(e)}"}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for crop prediction with selected model
    
    Expected JSON:
    {
        'N': nitrogen,
        'P': phosphorus,
        'K': potassium,
        'temperature': temp,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall,
        'model': 'dt' | 'rf' | 'knn' | 'svm'  (optional, defaults to 'rf')
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Get selected model (default to Random Forest)
        selected_model_key = data.get('model', 'rf')
        
        if selected_model_key != 'ensemble' and (selected_model_key not in models or models[selected_model_key] is None):
            selected_model_key = 'rf'
            
        # Extract features
        features = [float(data[field]) for field in feature_names]
        features_scaled = scaler.transform([features])[0]
        
        if selected_model_key == 'ensemble':
            preds = []
            for k in ['dt', 'rf', 'knn', 'svm']:
                if models[k] is not None:
                    preds.append(models[k].predict([features_scaled])[0])
            
            vote_counts = Counter(preds)
            most_common = vote_counts.most_common()
            
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                prediction = models['rf'].predict([features_scaled])[0] if models['rf'] else most_common[0][0]
            else:
                prediction = most_common[0][0]
                
            selected_confidence = float((vote_counts[prediction] / len(preds)) * 100) if len(preds) > 0 else 100.0
            alternatives = get_alternative_predictions(models['rf'], features, scaler, top_k=5) if models['rf'] else []
            selected_model = models.get('rf', None)  # For feature importance
        else:
            selected_model = models[selected_model_key]
            prediction = selected_model.predict([features_scaled])[0]
            alternatives = get_alternative_predictions(selected_model, features, scaler, top_k=5)
            
            if hasattr(selected_model, 'predict_proba'):
                probabilities = selected_model.predict_proba([features_scaled])[0]
                selected_confidence = float(np.max(probabilities) * 100)
            else:
                selected_confidence = 100.0
        
        # Get feature importance (if available)
        feature_importance = get_feature_importance(selected_model, feature_names)
        
        # Get all model predictions (for comparison)
        all_predictions = {}
        for model_key, model in models.items():
            if model is not None and model_key != 'ensemble':
                pred = model.predict([features_scaled])[0]
                alts = get_alternative_predictions(model, features, scaler, top_k=3)
                all_predictions[model_key] = {
                    'predicted_crop': str(pred),
                    'alternatives': alts,
                    'model_info': model_info[model_key]
                }
        
        # Manually add ensemble to all_predictions if selected, so frontend doesn't crash
        if selected_model_key == 'ensemble':
            all_predictions['ensemble'] = {
                'predicted_crop': str(prediction),
                'alternatives': alternatives,
                'model_info': model_info['ensemble']
            }
        
        crop_details = get_crop_details(prediction)
        crop_ranges = crop_feature_ranges.get(str(prediction).strip().lower(), None)

        # Prepare response
        response = {
            'success': True,
            'selected_model': selected_model_key,
            'selected_model_name': model_info[selected_model_key]['name'],
            'prediction': {
                'recommended_crop': str(prediction),
                'alternatives': alternatives,
                'confidence': selected_confidence,
                'feature_importance': feature_importance,
                'crop_details': crop_details,
                'crop_ranges': crop_ranges
            },
            'all_models': all_predictions,
            'input_summary': {
                'N': float(data['N']),
                'P': float(data['P']),
                'K': float(data['K']),
                'temperature': float(data['temperature']),
                'humidity': float(data['humidity']),
                'ph': float(data['ph']),
                'rainfall': float(data['rainfall'])
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Prediction error: {str(e)}"
        }), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """API endpoint for batch crop prediction using CSV upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded.'}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected.'}), 400

        model_key = request.form.get('model', 'rf')
        if model_key != 'ensemble' and (model_key not in models or models[model_key] is None):
            model_key = 'rf'

        df = pd.read_csv(uploaded_file)
        
        # Standardize column headers (strip whitespace)
        df.columns = [col.strip() for col in df.columns]
        
        # Case-insensitive column verification and mapping
        lowercase_cols = {col.lower(): col for col in df.columns}
        expected_lower = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_cols = [f for f in expected_lower if f not in lowercase_cols]
        
        if missing_cols:
            standard_names = {'n': 'N', 'p': 'P', 'k': 'K', 'temperature': 'temperature', 'humidity': 'humidity', 'ph': 'ph', 'rainfall': 'rainfall'}
            missing_report = [standard_names[c] for c in missing_cols]
            return jsonify({
                'success': False,
                'error': f"Missing required columns: {', '.join(missing_report)}"
            }), 400
        
        # Extract features and align column names case-sensitively to what's expected
        rename_dict = {lowercase_cols[f]: f_orig for f, f_orig in zip(expected_lower, feature_names)}
        inputs_df = df[[lowercase_cols[f] for f in expected_lower]].rename(columns=rename_dict)
        inputs = inputs_df.astype(float)
        
        scaled_inputs = scaler.transform(inputs)
        
        has_proba = False
        probabilities = None
        
        if model_key == 'ensemble':
            all_preds = []
            for k in ['dt', 'rf', 'knn', 'svm']:
                if models[k]:
                    all_preds.append(models[k].predict(scaled_inputs))
            
            predictions = []
            for i in range(len(scaled_inputs)):
                row_preds = [p[i] for p in all_preds]
                vote_counts = Counter(row_preds)
                most_common = vote_counts.most_common()
                if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                    predictions.append(models['rf'].predict([scaled_inputs[i]])[0] if models['rf'] else most_common[0][0])
                else:
                    predictions.append(most_common[0][0])
        else:
            model = models[model_key]
            # Batch inference: much faster than calling predict/predict_proba inside a loop row-by-row
            predictions = model.predict(scaled_inputs)
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(scaled_inputs)
                    has_proba = True
                except AttributeError:
                    pass
        
        # Precompute probabilities for the entire batch if the model supports it
        has_proba = False
        probabilities = None
        classes = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(scaled_inputs)
                classes = model.classes_
                has_proba = True
            except AttributeError:
                pass
        
        results = []
        top_k = 3
        for i, (idx, row) in enumerate(inputs.iterrows()):
            predicted_crop = str(predictions[i])
            
            # Retrieve alternative predictions using precomputed probabilities if available
            if has_proba:
                row_probs = probabilities[i]
                top_indices = np.argsort(row_probs)[-top_k:][::-1]
                alternatives = [
                    {
                        'crop': str(classes[idx_val]),
                        'confidence': float(row_probs[idx_val]) * 100
                    }
                    for idx_val in top_indices
                ]
            else:
                alternatives = [{'crop': predicted_crop, 'confidence': 100.0}]
                
            crop_details_data = get_crop_details(predicted_crop)
            results.append({
                'row': int(i) + 1,
                'input_values': row.to_dict(),
                'recommended_crop': predicted_crop,
                'alternatives': alternatives,
                'crop_details': crop_details_data
            })

        return jsonify({
            'success': True,
            'selected_model': model_key,
            'selected_model_name': model_info[model_key]['name'],
            'results': results
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Batch prediction error: {str(e)}"
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get information about all available models"""
    try:
        models_list = []
        for key, info in model_info.items():
            models_list.append({
                'key': key,
                'name': info['name'],
                'description': info['description'],
                'pros': info['pros'],
                'cons': info['cons'],
                'best_for': info['best_for']
            })
        
        return jsonify({
            'models': models_list,
            'default_model': 'rf'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info/<model_key>', methods=['GET'])
def get_model_info(model_key):
    """Get detailed information about a specific model"""
    if model_key in model_info:
        return jsonify(model_info[model_key]), 200
    else:
        return jsonify({'error': 'Model not found'}), 404


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Route not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  SMART FARMING SYSTEM - WEB APPLICATION (4 MODELS)".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    # Load models
    if load_models():
        print("Starting production WSGI server with Waitress...")
        print("\n✓ Access the application at: http://localhost:5000")
        print("✓ API Endpoints:")
        print("  - POST /api/predict - Make predictions")
        print("  - GET /api/models - List all models")
        print("  - GET /api/model-info/<model_key> - Get model details")
        print("✓ Press CTRL+C to stop the server\n")
        
        serve(app, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Cannot start application without trained models.")
        print("  Please run: python train.py")
