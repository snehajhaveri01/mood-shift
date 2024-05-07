import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import random


# Load the pre-trained model and transformers
def load_resources():
    # model = tf.keras.models.load_model('model_files/saved_model')
    # load h5 model from model_files/saved_model.h5
    model = tf.keras.models.load_model('src/model_files/saved_model.h5')
    # load tflite model
    # model = tf.lite.Interpreter(model_path='model_files/mood-shift.tflite')
    with open('src/model_files/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('src/model_files/text_transformer.pkl', 'rb') as f:
        text_transformer = pickle.load(f)
    return model, encoder, text_transformer


model, encoder, text_transformer = load_resources()


def load_data():
    with open("src/model_samples/activities.json", "r") as file:
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
    df['Activities'] = [activities] * len(df)
    return df


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


def predict_and_suggest_activities(aspect, mood, place, reason, sentiments):
    if mood.lower() in sentiments['Negative']:
        x_combined = preprocess_input(aspect, mood, place, reason)
        mood_index = predict_mood(x_combined)
        desired_mood = df['Desired_Mood'].iloc[mood_index]
        activities = df[df['Desired_Mood'] == desired_mood]['Activities'].iloc[0]
        random.shuffle(activities)
        return activities[:6]
    return "Keep up the good vibes!"


def load_sentiments(filename='src/model_samples/sentiments.json'):
    with open(filename, 'r') as file:
        return json.load(file)


sentiments = load_sentiments()
