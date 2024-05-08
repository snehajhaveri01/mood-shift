import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import random

# Path constants based on the project directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model_files')
SAMPLE_DIR = os.path.join(BASE_DIR, 'model_samples')


# Load the pre-trained model and transformers
def load_resources():
    try:
        # can load other model types here
        model_path = os.path.join(MODEL_DIR, 'saved_model.h5')  # Adjust path for different model types
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.") if loaded_model else print("Model not loaded.")

        encoder_path = os.path.join(MODEL_DIR, 'encoder.pkl')
        with open(encoder_path, 'rb') as f:
            loaded_encoder = pickle.load(f)

        text_transformer_path = os.path.join(MODEL_DIR, 'text_transformer.pkl')
        with open(text_transformer_path, 'rb') as f:
            loaded_text_transformer = pickle.load(f)

        return loaded_model, loaded_encoder, loaded_text_transformer
    except Exception as e:
        print("Error loading resources:", e)
        return None, None, None


model, encoder, text_transformer = load_resources()


def load_data():
    activities_path = os.path.join(SAMPLE_DIR, 'activities.json')
    with open(activities_path, "r") as file:
        activities = json.load(file)
    data = {
        'Aspect': ['Emotional', 'Mental', 'Physical', 'Emotional', 'Mental', 'Physical'],
        'Mood': ['sad', 'exhausted', 'weak', 'angry', 'stressed', 'numb'],
        'Place': ['home', 'work', 'personal', 'home', 'work', 'personal'],
        'Reasons': ['Mourned a personal achievement', 'Work deadlines and pressure', 'Lack of sleep',
                    'Received bad news', 'Conflict at work', 'Just finished a workout'],
        'Desired_Mood': ['happy', 'relaxed', 'flexible', 'focused', 'assertive', 'connected'],
    }
    loaded_df = pd.DataFrame(data)
    loaded_df['Activities'] = [activities] * len(loaded_df)
    return loaded_df


df = load_data()


def preprocess_input(aspect, mood, place, reason):
    x = pd.DataFrame({
        'Mood': [mood.lower()],
        'Aspect': [aspect.lower()],
        'Place': [place.lower()],
        'Reasons': [reason],
    })
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


def load_sentiments():
    sentiments_path = os.path.join(SAMPLE_DIR, 'sentiments.json')
    with open(sentiments_path, 'r') as file:
        return json.load(file)


sentiments = load_sentiments()
