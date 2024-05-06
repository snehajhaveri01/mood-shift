from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import random
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)


# Load the pre-trained model and transformers
def load_resources():
    custom_model = tf.keras.models.load_model('saved_model')
    with open('encoder.pkl', 'rb') as f:
        input_encoder = pickle.load(f)
    with open('text_transformer.pkl', 'rb') as f:
        entry_transformer = pickle.load(f)
    return custom_model, input_encoder, entry_transformer


model, encoder, text_transformer = load_resources()


# Load and prepare data
def load_data():
    data = {
        'Mood': ['sad', 'exhausted', 'weak', 'angry', 'stressed', 'numb'],
        'Aspect': ['Emotional', 'Mental', 'Physical', 'Emotional', 'Mental', 'Physical'],
        'Reasons': ['Mourned a personal achievement', 'Work deadlines and pressure', 'Lack of sleep',
                    'Received bad news', 'Conflict at work', 'Just finished a workout'],
        'Place': ['home', 'work', 'personal', 'home', 'work', 'personal'],
        'Desired_Mood': ['happy', 'relaxed', 'flexible', 'focused', 'assertive', 'connected'],
    }
    data_frame = pd.DataFrame(data)
    for column in ['Mood', 'Aspect', 'Place', 'Desired_Mood']:
        data_frame[column] = data_frame[column].str.lower()
    with open("activities.json", "r") as file:
        activities = json.load(file)
    data_frame['Activities'] = [activities] * len(data_frame)
    return data_frame


df = load_data()


# Preprocess input data
def preprocess_input(mood, aspect, reason, place):
    x = pd.DataFrame({
        'Mood': [mood.lower()],
        'Aspect': [aspect.lower()],
        'Reasons': [reason],
        'Place': [place.lower()]
    })
    reasons_text = text_transformer.transform(x['Reasons'])
    x_encoded = encoder.transform(x[['Mood', 'Aspect', 'Place']])
    x_combined = np.concatenate([x_encoded.toarray(), reasons_text.toarray()], axis=1)
    return x_combined


# Predict the desired mood
def predict_mood(x_combined):
    predictions = model.predict(x_combined)
    predicted_index = np.argmax(predictions)
    return df['Desired_Mood'].iloc[predicted_index]


def api_predict_mood():
    data = request.get_json()
    try:
        x_combined = preprocess_input(data['mood'], data['aspect'], data['reason'], data['place'])
        desired_mood = predict_mood(x_combined)
        return jsonify({'desired_mood': desired_mood})
    except KeyError as e:
        return jsonify({'error': f'Missing data for {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Retrieve suggested activities
def get_suggested_activities(desired_mood):
    activities = df[df['Desired_Mood'] == desired_mood]['Activities'].iloc[0]
    random.shuffle(activities)
    return activities[:6]


@app.route('/get_activities', methods=['POST'])
def api_get_activities():
    try:
        desired_mood = api_predict_mood().json['desired_mood']
        suggested_activities = get_suggested_activities(desired_mood)
        return jsonify({'activities': suggested_activities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5004)
