# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# import tensorflow as tf
# import json
# import random

# # Load and prepare data
# def load_and_prepare_data():
#     data = {
#         'User': ['John', 'Alice', 'Bob', 'Emily', 'David', 'Sara'],
#         'Mood': ['sad', 'exhausted', 'weak', 'angry', 'stressed', 'numb'],
#         'Aspect': ['Emotional', 'Mental', 'Physical', 'Emotional', 'Mental', 'Physical'],
#         'Reasons': ['Mourned a personal achievement', 'Work deadlines and pressure', 'Lack of sleep',
#                     'Received bad news', 'Conflict at work', 'Just finished a workout'],
#         'Place': ['home', 'work', 'personal', 'home', 'work', 'personal'],
#         'Desired_Mood': ['happy', 'relaxed', 'flexible', 'focused', 'assertive', 'connected'],
#     }
#     df = pd.DataFrame(data)

#     with open("activities.json", "r") as file:
#         activities = json.load(file)
#     df['Activities'] = [activities] * len(df)
#     return df

# # Build the model
# def build_and_compile_model(input_shape, num_classes):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(input_shape,)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Train the model
# def train_model(df):
#     text_transformer = Pipeline(steps=[
#         ('count_vectorizer', CountVectorizer())
#     ])
#     reasons_text = text_transformer.fit_transform(df['Reasons'])

#     encoder = OneHotEncoder(handle_unknown='ignore')
#     x_encoded = encoder.fit_transform(df[['Mood', 'Aspect', 'Place']])
#     x_combined = np.concatenate([x_encoded.toarray(), reasons_text.toarray()], axis=1)
#     y_encoded = pd.factorize(df['Desired_Mood'])[0]

#     model = build_and_compile_model(x_combined.shape[1], len(np.unique(y_encoded)))
#     model.fit(x_combined, y_encoded, epochs=10, batch_size=1)

#     # Save the model and preprocessors
#     model.save('saved_model')
#     with open('encoder.pkl', 'wb') as f:
#         pickle.dump(encoder, f)
#     with open('text_transformer.pkl', 'wb') as f:
#         pickle.dump(text_transformer, f)

#     return model, encoder, text_transformer

# # Predict mood and suggest activities
# def predict_and_suggest_activities(model, encoder, text_transformer, df, mood, aspect, reason, place):
#     input_data = preprocess_input(mood, aspect, reason, place, encoder, text_transformer)
#     predicted_index = model.predict(input_data).argmax()
#     desired_mood = df['Desired_Mood'].unique()[predicted_index]

#     if desired_mood in df['Desired_Mood'].values:
#         activities = df[df['Desired_Mood'] == desired_mood]['Activities'].iloc[0]
#         random.shuffle(activities)
#         return activities[:6]
#     return []

# # Preprocess input data
# def preprocess_input(mood, aspect, reason, place, encoder, text_transformer):
#     categorical_data = pd.DataFrame([[mood.lower(), aspect.lower(), place.lower()]],
#                                     columns=['Mood', 'Aspect', 'Place'])
#     categorical_encoded = encoder.transform(categorical_data).toarray()
#     reason_encoded = text_transformer.transform([reason]).toarray()
#     combined_input = np.concatenate([categorical_encoded, reason_encoded], axis=1)
#     return combined_input

# def load_sentiments(filename):
#     with open(filename, 'r') as file:
#         return json.load(file)
# # Load sentiments
# sentiments = load_sentiments('sentiments.json')

# # Main function to run training and prediction
# def main():
#     df = load_and_prepare_data()
#     model, encoder, text_transformer = train_model(df)

#     # Example input for prediction and activity suggestion
#     # aspect, mood, reason, place = 'emotional', 'sad', 'feeling low due to personal issues', 'home'
#     aspect, mood, reason, place = 'spiritual', 'empowered', 'aligning work-life balance', 'personal'
#     # Convert the mood to lowercase for case insensitivity
#     mood_lower = mood.lower()
    
#     emotion_sentiment = sentiments.get(mood_lower, None)
    
#     if mood_lower in sentiments['Positive']:
#             # Display appreciation message based on the positive sentiment
#             print(f"The selected emotion '{mood}' is positive. Keep up the good vibes!")
#     elif mood_lower in sentiments['Negative']:
#             print("Here are some suggested activities to improve your mood:")
#             activities = predict_and_suggest_activities(model, encoder, text_transformer, df, mood_lower, aspect, reason, place)
#             for activity in activities:
#                 print(activity["title"])
#                 print(activity["link"])
#     else:
#         print("Sentiment analysis not available for the selected mood.")

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import json
import pickle

# Load and prepare data
def load_and_prepare_data():
    df = pd.read_csv('data.csv')

    with open("activities.json", "r") as file:
        activities = json.load(file)
    df['Activities'] = [activities] * len(df)
    return df

# Train the model
def train_model(df):
    text_transformer = Pipeline(steps=[
        ('count_vectorizer', CountVectorizer())
    ])
    reasons_text = text_transformer.fit_transform(df['Reasons'])

    encoder = OneHotEncoder(handle_unknown='ignore')
    x_encoded = encoder.fit_transform(df[['Mood', 'Aspect', 'Place']])
    x_combined = np.concatenate([x_encoded.toarray(), reasons_text.toarray()], axis=1)

    # Save the preprocessors
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('text_transformer.pkl', 'wb') as f:
        pickle.dump(text_transformer, f)

    return encoder, text_transformer

# Recommend activities
def recommend_activities(mood, aspect, reason, place):
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('text_transformer.pkl', 'rb') as f:
        text_transformer = pickle.load(f)

    input_data = preprocess_input(mood, aspect, reason, place, encoder, text_transformer)
# 
    # Load activities
    with open("activities.json", "r") as file:
        activities_data = json.load(file)

    recommended_activities = []
    for activity in activities_data:
       if aspect.lower() in activity['aspect'].lower():
            recommended_activities.append(activity)

    return recommended_activities[:6]

# Preprocess input data
def preprocess_input(mood, aspect, reason, place, encoder, text_transformer):
    categorical_data = pd.DataFrame([[mood.lower(), aspect.lower(), place.lower()]],
                                    columns=['Mood', 'Aspect', 'Place'])
    categorical_encoded = encoder.transform(categorical_data).toarray()
    reason_encoded = text_transformer.transform([reason]).toarray()
    combined_input = np.concatenate([categorical_encoded, reason_encoded], axis=1)
    return combined_input

def main():
    df = load_and_prepare_data()
    encoder, text_transformer = train_model(df)

    # Example user inputs
    aspect = 'spiritual'
    mood = 'empowered'
    reason = 'aligning work-life balance'
    place = 'personal'

    # Recommend activities based on user inputs
    activities = recommend_activities(mood, aspect, reason, place)
    if activities:
        print("Suggested activities to improve mood:")
        for activity in activities:
            print(activity["Title"])
            print(activity["Link"])
    else:
        print("No activities found for the provided inputs.")

if __name__ == "__main__":
    main()
