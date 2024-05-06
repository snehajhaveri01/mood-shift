import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import tensorflow as tf
import json
import random

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
tf.keras.models.save_model(model, 'mood_shift_saved_model')

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

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
num_episodes = 1000
num_states = len(df)  # Number of states (assuming each row in DataFrame is a state)
num_actions = 6  # Number of actions (assuming 6 activities per state)

# Initialize Q-table with zeros
q_table = np.zeros((num_states, num_actions))

# Integration with Sentiment Analysis Model
def perform_sentiment_analysis(mood, reason, place):
    # Implement sentiment analysis logic here
    # This function should return the sentiment (positive/negative/neutral)
    sentiment = "positive"  # Placeholder, replace with actual sentiment analysis result
    return sentiment

# Environment Dynamics and Reward Structure
def get_reward(selected_activity, desired_mood):
    # Implement reward logic here
    # Calculate reward based on how well the selected activity matches the desired mood
    if selected_activity == desired_mood:
        return 1  # Positive reward for matching activity
    else:
        return 0  # No reward for mismatched activity

# Termination Condition
def is_episode_finished():
    # Implement termination condition here
    # Return True when episode should be terminated, False otherwise
    # Example: Terminate after a fixed number of steps or when certain criteria are met
    pass

# Interaction with User Input and Sentiment Analysis
for episode in range(num_episodes):
    state = np.random.randint(0, num_states)  # Initial state (random)
    done = False
    while not done:
        mood = input("Enter your current mood: ").lower()
        reason = input("Enter the reason for your current mood: ").lower()
        place = input("Enter the place you are currently in: ").lower()
        
        sentiment = perform_sentiment_analysis(mood, reason, place)
        desired_mood = predict_desired_mood(mood, reason, place, encoder, text_transformer)
        
        action = np.argmax(q_table[state, :])  # Select action with highest Q-value
        selected_activity = df['Activities'].iloc[state][action]
        reward = get_reward(selected_activity, desired_mood)
        
        next_state = np.random.randint(0, num_states)  # For demonstration purposes
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        
        state = next_state
        
        done = is_episode_finished()

# Saving and Loading Q-Table (optional)
np.save("q_table.npy", q_table)
# To load the Q-table: q_table = np.load("q_table.npy")
