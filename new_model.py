import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import random

# Load emotions from text file
def load_emotions(file_path):
    with open(file_path, "r") as file:
        emotions = file.readlines()
    return [emotion.strip() for emotion in emotions]

# Load positive and negative emotions
emotions = load_emotions("emotions.txt")

# Function to determine if the input text is positive or negative
def classify_emotion(text):
    positive_count = sum(1 for word in text.split() if word in emotions)
    negative_count = sum(1 for word in text.split() if word in emotions)
    return positive_count > negative_count

# Load activities from text file
def load_activities(file_path):
    with open(file_path, "r") as file:
        activities = file.readlines()
    return [activity.strip() for activity in activities]

# Sample Data
data = {
    'Mood': ['sad', 'stressed', 'tired', 'depressed', 'angry', 'relaxed'],
    'Aspect': ['Emotional', 'Mental', 'Physical', 'Emotional', 'Mental', 'Physical'],
    'Reasons': ['Mourned a personal achievement', 'Work deadlines and pressure', 'Lack of sleep',
                'Received bad news', 'Conflict at work', 'Just finished a workout'],
}

df = pd.DataFrame(data)

# Normalize categorical data
for column in ['Mood', 'Aspect']:
    df[column] = df[column].str.lower()

# TensorFlow Model for Text Classification
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10000,)),  # Assuming a vocabulary of 10,000 words
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train text classification model
vectorizer = CountVectorizer(max_features=10000)
X_text = vectorizer.fit_transform(df['Reasons'])
y_text = np.array([0 if mood in ['sad', 'stressed', 'tired', 'depressed', 'angry'] else 1 for mood in df['Mood']])
model.fit(X_text.toarray(), y_text, epochs=10, batch_size=1)

# Load activities
activities = load_activities("activities.txt")

# Function to recommend activities based on mood classification
def recommend_activities(is_positive_mood):
    activities = activities if is_positive_mood else activities
    print("Recommended activities:")
    random.shuffle(activities)
    for activity in activities[:6]:
        print(f"- {activity}")

# Function to handle user input
def handle_user_input():
    mood, aspect, reason = input("Enter your current mood: ").lower(), input("Enter your aspect: ").lower(), input("Enter the reason for your current mood: ").lower()
    is_positive_mood = classify_emotion(reason)
    recommend_activities(is_positive_mood)

handle_user_input()
