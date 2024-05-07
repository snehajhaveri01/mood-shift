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
    model = tf.keras.models.load_model('saved_model')
    # Load model in H5 format
    # model = tf.keras.models.load_model('saved_model.h5')

    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('text_transformer.pkl', 'rb') as f:
        text_transformer = pickle.load(f)
    return model, encoder, text_transformer


model, encoder, text_transformer = load_resources()


# Load and prepare data
def load_data():
    with open("activities.json", "r") as file:
        activities = json.load(file)
    data = {
        'Aspect': ['Emotional', 'Mental', 'Physical', 'Emotional', 'Mental', 'Physical'],
        'Mood': ['sad', 'exhausted', 'weak', 'angry', 'stressed', 'numb'],
        'Place': ['home', 'work', 'personal', 'home', 'work', 'personal'],
        'Reasons': ['Mourned a personal achievement', 'Work deadlines and pressure', 'Lack of sleep',
                    'Received bad news', 'Conflict at work', 'Just finished a workout'],
        'Desired_Mood': ['happy', 'relaxed', 'flexible', 'focused', 'assertive', 'connected'],
    }
    df = pd.DataFrame(data)
    df = df.map(str.lower)
    df['Activities'] = [activities] * len(df)
    return df


df = load_data()


def preprocess_input(aspect, mood, place, reason):
    # Create a DataFrame with the exact same column order and naming convention as during training
    x = pd.DataFrame({
        'Mood': [mood.lower()],
        'Aspect': [aspect.lower()],
        'Place': [place.lower()],
        'Reasons': [reason],  # Assuming reasons_text uses this column exactly as is
    })

    # Apply text transformer and encoder with the exact order
    reasons_text = text_transformer.transform(x['Reasons'])
    x_encoded = encoder.transform(x[['Mood', 'Aspect', 'Place']])
    x_combined = np.concatenate([x_encoded.toarray(), reasons_text.toarray()], axis=1)
    return x_combined


def predict_mood(x_combined):
    predictions = model.predict(x_combined)
    return np.argmax(predictions)


def predict_and_suggest_activities(aspect, mood, place, reason):
    if mood.lower() in sentiments['Negative']:
        x_combined = preprocess_input(aspect, mood, place, reason)
        mood_index = predict_mood(x_combined)
        desired_mood = df['Desired_Mood'].iloc[mood_index]
        activities = df[df['Desired_Mood'] == desired_mood]['Activities'].iloc[0]
        random.shuffle(activities)
        return activities[:6]
    return "Keep up the good vibes!"


# create health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/')
def index():
    return 'Welcome to Mood Shift!'


@app.route('/get_activities', methods=['POST'])
def api_get_activities():
    data = request.get_json()
    if not data or any(key not in data for key in ['mood', 'aspect', 'place', 'reason']):
        return jsonify({'error': 'Missing required data in request'}), 400

    try:
        # Ensure data is in the correct format and order
        x_combined = preprocess_input(data['aspect'], data['mood'], data['place'], data['reason'])
        mood_index = predict_mood(x_combined)
        desired_mood = df['Desired_Mood'].iloc[mood_index]

        # Check if the mood is negative and suggest activities
        if data['mood'].lower() in sentiments['Negative']:
            activities = predict_and_suggest_activities(data['aspect'], data['mood'], data['place'], data['reason'])
            return jsonify({'activities': activities})
        else:
            return jsonify({'message': "Keep up the good vibes!"})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def load_sentiments(filename='sentiments.json'):
    with open(filename, 'r') as file:
        return json.load(file)


sentiments = load_sentiments()

if __name__ == '__main__':
    app.run(debug=False)
