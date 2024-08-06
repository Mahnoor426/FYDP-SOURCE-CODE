# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import numpy as np
# import pandas as pd
# import logging

# app = Flask(__name__)
# CORS(app)  # Enable CORS
# logging.basicConfig(level=logging.DEBUG)  # Set logging level to debug



# # Load the trained models and other necessary objects
# with open('crop_pipeline.pkl', 'rb') as f:
#     forest_crop_model = pickle.load(f)
# with open('yield_pipeline.pkl', 'rb') as f:
#     forest_yield_model = pickle.load(f)
# with open('scaler (1).pkl', 'rb') as f:
#     scaler = pickle.load(f)
# with open('label_encoders (1).pkl', 'rb') as f:
#     label_encoders = pickle.load(f)


# # # Load the trained models and other necessary objects
# # with open('forest_crop_model.pkl', 'rb') as f:
# #     forest_crop_model = pickle.load(f)
# # with open('forest_yield_model.pkl', 'rb') as f:
# #     forest_yield_model = pickle.load(f)
# # with open('scaler.pkl', 'rb') as f:
# #     scaler = pickle.load(f)
# # with open('label_encoders.pkl', 'rb') as f:
# #     label_encoders = pickle.load(f)


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         logging.debug(f"Received data: {data}")

#         # Validate and convert input data
#         try:
#             division_name = data['division_name']
#             district_name = data['district_name']
#             year = float(data['year'])
#             temperature_max = float(data['temperature_max'])
#             temperature_min = float(data['temperature_min'])
#             rainfall_in_mm = float(data['rainfall_in_mm'])
#             area = float(data['area'])
#             ph = float(data['ph'])
            
#         except (KeyError, ValueError) as e:
#             logging.error(f"Data validation error: {e}")
#             return jsonify({'error': f"Invalid input data: {e}"}), 400

#         # Encode the categorical variables
#         try:
#             division_name_encoded = label_encoders['Division_Name'].transform([division_name])[0]
#             district_name_encoded = label_encoders['District_Name'].transform([district_name])[0]
#         except Exception as e:
#             logging.error(f"Encoding error: {e}")
#             return jsonify({'error': f"Encoding error: {e}"}), 400

#         # Prepare the features as a DataFrame with correct column names
#         features = pd.DataFrame({
#             'Division_Name': [division_name_encoded],
#             'District_Name': [district_name_encoded],
#             'Crop_Year': [year],  # Assuming this was the original name used for year
#             'Temperature (MAX)': [temperature_max],  # Adjust based on original names
#             'Temperature (MIN)': [temperature_min],  # Adjust based on original names
#             'Rainfall': [rainfall_in_mm],  # Adjust based on original names
#             'area': [area],
#             'pH': [ph] # Adjust based on original names
#         })

#         logging.debug(f"Prepared features: {features}")

#         # Scale the features
#         try:
#             # Ensure features columns match X.columns order and names
#             features_scaled = scaler.transform(features)
#         except Exception as e:
#             logging.error(f"Scaling error: {e}")
#             return jsonify({'error': f"Scaling error: {e}"}), 400

#         # Make a prediction for the crop
#         try:
#             crop_prediction_encoded = forest_crop_model.predict(features_scaled)[0]
#             crop_prediction = label_encoders['Crop'].inverse_transform([crop_prediction_encoded])[0]
#         except Exception as e:
#             logging.error(f"Crop prediction error: {e}")
#             return jsonify({'error': f"Crop prediction error: {e}"}), 400

#         # Make a prediction for the yield
#         try:
#             yield_prediction = forest_yield_model.predict(features_scaled)[0]
#         except Exception as e:
#             logging.error(f"Yield prediction error: {e}")
#             return jsonify({'error': f"Yield prediction error: {e}"}), 400

#         return jsonify({'predicted_crop': crop_prediction, 'predicted_yield': yield_prediction})

#     except Exception as e:
#         logging.error(f"Unhandled error: {e}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)





from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS
logging.basicConfig(level=logging.DEBUG)  # Set logging level to debug

# Load the trained models and other necessary objects
with open('crop_pipeline.pkl', 'rb') as f:
    crop_pipeline = pickle.load(f)
with open('yield_pipeline.pkl', 'rb') as f:
    yield_pipeline = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

def apply_custom_rule(pH):
    if 6.0 <= pH <= 6.3:
        return 'rape seed'
    elif 6.3 < pH <= 6.5:
        return 'barley'
    elif 6.5 < pH <= 7.0:
        return 'wheat'
    elif 7.0 < pH <= 7.1:
        return 'fodder crop'
    elif 7.1 < pH <= 7.5:
        return 'mustard seed'
    elif 7.5 < pH <= 8.5:
        return 'gram'
    else:
        return "unknown"

def adjust_yield_based_on_area(predicted_yield, area, rainfall, temperature, crop):
    # Adjust yield based on area
    if area < 100:
        adjusted_yield = predicted_yield * 0.3
    elif 100 <= area < 200:
        adjusted_yield = predicted_yield * 0.6
    elif 200 <= area < 300:
        adjusted_yield = predicted_yield * 0.8
    elif 300 <= area < 400:
        adjusted_yield = predicted_yield * 1.0
    elif 400 <= area < 500:
        adjusted_yield = predicted_yield * 1.2
    elif 500 <= area < 600:
        adjusted_yield = predicted_yield * 1.4
    else:
        adjusted_yield = predicted_yield * 1.6

    # Adjust yield based on rainfall
    if crop in ['barley', 'rape seed', 'fodder crop', 'wheat']:
        if rainfall < 200:
            adjusted_yield *= 0.5
        elif 200 <= rainfall < 300:
            adjusted_yield *= 0.8
        elif 300 <= rainfall < 400:
            adjusted_yield *= 1.0
        elif 400 <= rainfall < 500:
            adjusted_yield *= 1.2
        else:
            adjusted_yield *= 1.4
    elif crop in ['chickpea', 'mustard seed']:
        if rainfall < 300:
            adjusted_yield *= 0.6
        elif 300 <= rainfall < 400:
            adjusted_yield *= 0.9
        elif 400 <= rainfall < 500:
            adjusted_yield *= 1.1
        else:
            adjusted_yield *= 1.3

    # Adjust yield based on temperature
    if crop in ['wheat', 'barley', 'rape seed', 'chickpea', 'mustard seed']:
        if temperature < 5:
            adjusted_yield *= 0.3
        elif 5 <= temperature < 10:
            adjusted_yield *= 0.6
        elif 10 <= temperature < 15:
            adjusted_yield *= 0.8
        elif 15 <= temperature < 20:
            adjusted_yield *= 1.0
        elif 20 <= temperature < 25:
            adjusted_yield *= 1.2
        elif 25 <= temperature < 30:
            adjusted_yield *= 1.4
        else:
            adjusted_yield *= 1.6
    elif crop in ['fodder crop', 'gram']:
        if temperature < 10:
            adjusted_yield *= 0.4
        elif 10 <= temperature < 15:
            adjusted_yield *= 0.7
        elif 15 <= temperature < 20:
            adjusted_yield *= 0.9
        elif 20 <= temperature < 25:
            adjusted_yield *= 1.1
        elif 25 <= temperature < 30:
            adjusted_yield *= 1.3
        else:
            adjusted_yield *= 1.5

    return adjusted_yield

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")

        # Validate and convert input data
        try:
            division_name = data['division_name']
            district_name = data['district_name']
            year = float(data['year'])
            temperature_max = float(data['temperature_max'])
            temperature_min = float(data['temperature_min'])
            rainfall_in_mm = float(data['rainfall_in_mm'])
            area = float(data['area'])
            ph = float(data['ph'])
        except (KeyError, ValueError) as e:
            logging.error(f"Data validation error: {e}")
            return jsonify({'error': f"Invalid input data: {e}"}), 400

        # Apply custom rule for pH
        custom_rule_result = apply_custom_rule(ph)

        if custom_rule_result == "unknown":
            return jsonify({'predicted_crop': "No Rabi Crop grow in this pH value", 'predicted_yield': "No Yield"})

        elif custom_rule_result != "unknown":
            crop_prediction = custom_rule_result

            # Define a default features_scaled for yield prediction when custom rule applies
            features = pd.DataFrame({
                'Division_Name': [0],  # Placeholder value
                'District_Name': [0],  # Placeholder value
                'Crop_Year': [year],
                'Temperature (MAX)': [temperature_max],
                'Temperature (MIN)': [temperature_min],
                'Rainfall': [rainfall_in_mm],
                'area': [area],
                'pH': [ph]
            })

            logging.debug(f"Prepared features for custom rule: {features}")

            # Scale the features
            try:
                features_scaled = scaler.transform(features)
            except Exception as e:
                logging.error(f"Scaling error: {e}")
                return jsonify({'error': f"Scaling error: {e}"}), 400

        else:
            # Encode the categorical variables
            try:
                division_name_encoded = label_encoders['Division_Name'].transform([division_name])[0]
                district_name_encoded = label_encoders['District_Name'].transform([district_name])[0]
            except Exception as e:
                logging.error(f"Encoding error: {e}")
                return jsonify({'error': f"Encoding error: {e}"}), 400

            # Prepare the features
            features = pd.DataFrame({
                'Division_Name': [division_name_encoded],
                'District_Name': [district_name_encoded],
                'Crop_Year': [year],
                'Temperature (MAX)': [temperature_max],
                'Temperature (MIN)': [temperature_min],
                'Rainfall': [rainfall_in_mm],
                'area': [area],
                'pH': [ph]
            })

            logging.debug(f"Prepared features: {features}")

            # Scale the features
            try:
                features_scaled = scaler.transform(features)
            except Exception as e:
                logging.error(f"Scaling error: {e}")
                return jsonify({'error': f"Scaling error: {e}"}), 400

            # Predict crop
            try:
                crop_prediction_encoded = crop_pipeline.named_steps['svm_crop'].predict(features_scaled)[0]
                crop_prediction = label_encoders['Crop'].inverse_transform([crop_prediction_encoded])[0]
            except Exception as e:
                logging.error(f"Crop prediction error: {e}")
                return jsonify({'error': f"Crop prediction error: {e}"}), 400



        # Predict yield
        try:
            yield_prediction = yield_pipeline.predict(features_scaled)[0]
            yield_prediction = adjust_yield_based_on_area(yield_prediction, area, rainfall_in_mm, (temperature_max + temperature_min) / 2, crop_prediction)
        except Exception as e:
            logging.error(f"Yield prediction error: {e}")
            return jsonify({'error': f"Yield prediction error: {e}"}), 400

        return jsonify({'predicted_crop': crop_prediction, 'predicted_yield': yield_prediction})

    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

