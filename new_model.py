import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import tensorflow as tf
import random

# Sample Data
data = {
    'User': ['John', 'Alice', 'Bob', 'Emily', 'David', 'Sara'],
    'Mood': ['sad', 'stressed', 'tired', 'depressed', 'angry', 'relaxed'],
    'Aspect': ['Emotional', 'Mental', 'Physical', 'Emotional', 'Mental', 'Physical'],
    'Reasons': ['Mourned a personal achievement', 'Work deadlines and pressure', 'Lack of sleep',
                'Received bad news', 'Conflict at work', 'Just finished a workout'],
    'Place': ['home', 'work', 'personal', 'home', 'work', 'personal'],
    'Desired_Mood': ['joyful', 'motivated', 'energetic', 'happy', 'calm', 'productive']
}

df = pd.DataFrame(data)

# Normalize categorical data
for column in ['Mood', 'Aspect', 'Place', 'Desired_Mood']:
    df[column] = df[column].str.lower()

# Read activities from a text file
with open("activities.txt", "r") as file:
    activities = file.readlines()
activities = [activity.strip() for activity in activities]

# Add activities to the dataframe
df['Activities'] = [activities] * len(df)

# Preprocessing
X = df[['Mood', 'Aspect', 'Reasons', 'Place']]
y = df['Desired_Mood']

# Text Preprocessing
text_transformer = Pipeline(steps=[
    ('count_vectorizer', CountVectorizer())
])
reasons_text = text_transformer.fit_transform(X['Reasons'])

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
categorical_features = ['Mood', 'Aspect', 'Place']
X_encoded = encoder.fit_transform(X[categorical_features])

# Combine inputs
X_combined = np.concatenate([X_encoded.toarray(), reasons_text.toarray()], axis=1)

# TensorFlow Model for Multi-Class Classification
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_combined.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(y.unique()), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Encode the labels for multi-class classification
y_encoded = pd.factorize(y)[0]

# Train the model
model.fit(X_combined, y_encoded, epochs=10, batch_size=1)

# model.save('mood-shift.keras')
tf.keras.models.save_model(model, 'mood-shift_saved_model')


def predict_desired_mood(mood, aspect, reason, place):
    # input_data = text_transformer.transform([reason]).toarray()
    input_data = preprocess_input(mood, aspect, reason, place, encoder, text_transformer)
    predicted_index = model.predict(input_data).argmax()
    return y.unique()[predicted_index]

# Function to preprocess input data
def preprocess_input(mood, aspect, reason, place, encoder, text_transformer):
    categorical_data = pd.DataFrame([[mood, aspect, place]], columns=['Mood', 'Aspect', 'Place'])
    categorical_encoded = encoder.transform(categorical_data).toarray()
    reason_encoded = text_transformer.transform([reason]).toarray()
    combined_input = np.concatenate([categorical_encoded, reason_encoded], axis=1)
    return combined_input


with open("emotions.txt", "r") as file:
    emotions = file.readlines()
emotions = [emotion.strip() for emotion in emotions]

def handle_user_input():
    aspects = ['Mental', 'Physical', 'Emotional', 'Spiritual']
    print("Select an aspect:")
    for index, aspect in enumerate(aspects, start=1):
        print(f"{index}. {aspect}")
    
    # Prompt the user to select an aspect
    aspect_choice = input("Enter the number corresponding to your chosen aspect: ")
    try:
        aspect_index = int(aspect_choice) - 1
        aspect = aspects[aspect_index]
    except (ValueError, IndexError):
        print("Invalid aspect selection.")
        return
    
    mood = input("Enter your current mood: ").lower()
    reason = input("Enter the reason for your current mood: ")
    place = input("Enter the place you are currently in: ")
    

    desired_mood = predict_desired_mood(mood, aspect, reason, place)
    
    # Determine activities based on desired mood
    activities = df[df['Desired_Mood'] == desired_mood]['Activities'].iloc[0]
    
    if activities:
        # Shuffle the list
        random.shuffle(activities)
        # Print first six activities
        for activity in activities[:min(6, len(activities))]:
            print(f"- {activity.strip()}")
    else:
        print("No activities found for the selected mood.")

handle_user_input()



