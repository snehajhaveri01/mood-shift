import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import tensorflow as tf
import json

# Sample Data
data = {
    'User': ['John', 'Alice', 'Bob', 'Emily', 'David', 'Sara'],
    'Mood': ['sad', 'exhausted', 'weak', 'angry', 'stressed', 'numb'],
    'Aspect': ['Emotional', 'Mental', 'Physical', 'Emotional', 'Mental', 'Physical'],
    'Reasons': ['Mourned a personal achievement', 'Work deadlines and pressure', 'Lack of sleep',
                'Received bad news', 'Conflict at work', 'Just finished a workout'],
    'Place': ['home', 'work', 'personal', 'home', 'work', 'personal'],
    'Desired_Mood': ['happy', 'relaxed', 'flexible', 'focused', 'assertive', 'connected'],
}

df = pd.DataFrame(data)

# Normalize categorical data
for column in ['Mood', 'Aspect', 'Place', 'Desired_Mood']:
    df[column] = df[column].str.lower()

# Read activities from a JSON file
with open("updated_activities.json", "r") as file:
    activities = json.load(file)

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

# Function to preprocess input data
def preprocess_input(mood, reason, place, encoder, text_transformer):
    categorical_data = pd.DataFrame([[mood, None, place]], columns=['Mood', 'Aspect', 'Place'])
    categorical_encoded = encoder.transform(categorical_data).toarray()
    reason_encoded = text_transformer.transform([reason]).toarray()
    combined_input = np.concatenate([categorical_encoded, reason_encoded], axis=1)
    return combined_input

def predict_desired_mood(mood, reason, place, encoder, text_transformer):
    input_data = preprocess_input(mood, reason, place, encoder, text_transformer)
    predicted_index = model.predict(input_data).argmax()
    return y.unique()[predicted_index]

with open("emotions.json", "r") as file:
    emotions = json.load(file)

# Read activities from a JSON file
with open("sentiments.json", "r") as file:
    sentiments = json.load(file)

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
    
    # Load emotions based on the aspect selection
    with open("emotions.json", "r") as file:
        emotions_dict = json.load(file)
    if aspect.lower() in emotions_dict:
        emotions = emotions_dict[aspect.lower()]
        # Provide emotions based on the aspect selection
        print("Select an emotion:")
        for index, emotion in enumerate(emotions, start=1):
            print(f"{index}. {emotion}")
        emotion_choice = input("Enter the number corresponding to your chosen emotion: ")
        try:
            emotion_index = int(emotion_choice) - 1
            mood = emotions[emotion_index]
            get_sentiment(mood, sentiments)
        except (ValueError, IndexError):
            print("Invalid emotion selection.")
            return
    else:
        print("No emotions found for the selected aspect.")
        return
  

def get_sentiment(mood, sentiments):
    reason = input("Enter the reason for your current mood: ")
    place = input("Enter the place you are currently in: ")
    mood = mood.lower()
    if mood in sentiments['Positive']:
        print('Hey, I am so happy that you are feeling good, hoping that you continue to be good')
        return  # Return after printing the message
    elif mood in sentiments['Negative']:
        desired_mood = predict_desired_mood(mood, reason, place, encoder, text_transformer)
        
        # Determine activities based on desired mood
        activities = df[df['Desired_Mood'] == desired_mood]['Activities'].iloc[0]
        
        if activities:
            # Print activities in the specified format
            num_displayed_activities = 0
            for activity in activities:
                if num_displayed_activities == 6:
                    break
                title = activity.get("title", "")
                print(f"{num_displayed_activities+1}. {title}")
                num_displayed_activities += 1
            
            activity_choice = input("Enter the number corresponding to your chosen activity: ")
            try:
                activity_index = int(activity_choice) - 1
                selected_activity = activities[activity_index]
                title = selected_activity.get("title", "")
                instructions = selected_activity.get("instructions", "")
                link = selected_activity.get("link", "")
                print(f"Title: {title}")
                print(f"Instructions: {instructions}")
                print(f"Link: {link}")
            except (ValueError, IndexError):
                print("Invalid activity selection.")
        else:
            print("No activities found for the selected mood.")



handle_user_input()




