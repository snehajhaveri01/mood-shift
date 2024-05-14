# import pandas as pd
# from sklearn.cluster import KMeans
# import pickle
# import json
# # import random

# # Load the data
# df = pd.read_csv('data.csv')

# # Exclude 'User' column
# df_for_clustering = df.drop('User', axis=1)

# # Perform one-hot encoding for categorical variables
# df_encoded = pd.get_dummies(df_for_clustering)

# # Create the KMeans model
# kmeans = KMeans(n_clusters=3)

# # Fit the model to the data
# kmeans.fit(df_encoded)

# # # # Load the saved model and preprocessors for recommending activities
# # with open('saved_model', 'rb') as f:
# #     model = pickle.load(f)
# with open('encoder.pkl', 'rb') as f:
#     encoder = pickle.load(f)
# with open('text_transformer.pkl', 'rb') as f:
#     text_transformer = pickle.load(f)

# # Load activities
# with open("activities.json", "r") as file:
#     activities_data = json.load(file)

# # Function to recommend activities based on user inputs
# def recommend_activities(aspect, mood, reason, place):
#      # Convert user input to match the format used during training
#     aspect = aspect.capitalize()
#     mood = mood.capitalize()
#     reason = reason.capitalize()
#     place = place.capitalize()

#     user_data = pd.DataFrame([[mood.lower(), aspect.lower(), reason.lower(), place.lower()]],
#                              columns=['Mood', 'Aspect', 'Reasons', 'Place'])
    
#      # Encode user data to match the format used during training
#     user_encoded = pd.get_dummies(user_data)

#      # Predict cluster for user data
#     user_cluster = kmeans.predict(user_encoded)[0]
    
#     # Filter cluster data based on predicted cluster
#     cluster_data = df_encoded[kmeans.labels_ == user_cluster]
    
#     # Predict and suggest activities for the filtered cluster data
#     activities = predict_and_suggest_activities(encoder, text_transformer, cluster_data)
#     return activities

# # Predict mood and suggest activities
# def predict_and_suggest_activities(model, encoder, text_transformer, df):
#     input_data = preprocess_input(df, encoder, text_transformer)
#     predicted_index = model.predict(input_data).argmax()
#     predicted_mood = df['Mood'].unique()[predicted_index]

#     # Filter activities based on mood, aspect, reason, and place
#     filtered_activities = [activity for activity in activities_data.get(predicted_mood, [])
#                            if activity['aspect'] == df['Aspect'].iloc[0] and
#                            activity['mood'] == df['Reasons'].iloc[0] and
#                            activity['reason'] == df['Reasons'].iloc[0] and
#                            activity['place'] == df['Place'].iloc[0]]
    
#     # random.shuffle(filtered_activities)
#     return filtered_activities[:6]

# # Preprocess input data
# def preprocess_input(df, encoder, text_transformer):
#     categorical_encoded = encoder.transform(df[['Mood', 'Aspect', 'Place']])
#     reasons_text = text_transformer.transform(df['Reasons'])

#     combined_input = pd.concat([pd.DataFrame(categorical_encoded.toarray()), pd.DataFrame(reasons_text.toarray())], axis=1)
#     return combined_input

# # Example user inputs
# aspect = 'spiritual'
# mood = 'empowered'
# reason = 'aligning work-life balance'
# place = 'personal'

# # Recommend activities based on user inputs
# activities = recommend_activities(aspect, mood, reason, place)
# if activities:
#     print("Suggested activities to improve mood:")
#     for activity in activities:
#         print(activity["title"])
#         print(activity["link"])
# else:
#     print("No activities found for the provided inputs.")


import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Step 1: Load the Data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Step 2: Data Preprocessing (No extensive preprocessing needed)

# Step 3: Feature Extraction (Extract aspect labels and instructions)
def extract_aspect_labels(data):
    aspect_labels = []
    instructions = []
    for item in data:
        aspect_labels.extend([item['aspect']] * len(item['instructions']))
        instructions.extend(item['instructions'])
    return aspect_labels, instructions


# Step 4: Clustering
def cluster_activities(tfidf_matrix, aspect_labels, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(tfidf_matrix)
    return kmeans

# Step 5: Recommendation
def recommend_activity(user_input, kmeans_model, aspect_labels, instructions, tfidf_vectorizer, data):
    # Extract user input
    aspect = user_input['Aspect']
    mood = user_input['Mood']
    reasons = user_input['Reasons']
    place = user_input['Place']
    
    # Filter instructions based on aspect
    aspect_indices = [i for i, label in enumerate(aspect_labels) if label.lower() == aspect.lower()]
    
    # If there are exactly 6 activities matching the aspect, use them
    if len(aspect_indices) == 6:
        recommended_indices = aspect_indices
    else:
        # Transform aspect instructions using the same TF-IDF vectorizer instance
        aspect_tfidf_matrix = tfidf_vectorizer.transform([instructions[i] for i in aspect_indices])
        
        # Predict cluster for the aspect instructions
        predicted_cluster = kmeans_model.predict(aspect_tfidf_matrix)
        
        # Find most common cluster for the aspect
        common_cluster = np.bincount(predicted_cluster).argmax()
        
        # Get recommendations from the most common cluster
        recommended_indices = np.where(predicted_cluster == common_cluster)[0]
        
        # Ensure we recommend exactly 6 activities
        recommended_indices = recommended_indices[:6]

    # Get titles and instructions of recommended activities
    recommended_titles = [data[i]['title'] for i in recommended_indices]
    
    return recommended_titles 


# Example usage
def main():
    # Step 1: Load the Data
    file_path = 'activities.json'
    data = load_data(file_path)
    
    # Step 2: Data Preprocessing (Not needed for this example)
    
    # Step 3: Feature Extraction
    aspect_labels, instructions = extract_aspect_labels(data)
    
    # Step 4: Clustering
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(instructions)
    kmeans_model = cluster_activities(tfidf_matrix, aspect_labels)
    
    # Step 5: Recommendation
    user_input = {
        'Aspect': 'Emotional',
        'Mood': 'Lonely',
        'Reasons': 'Not feeling good and missing my favorite person',
        'Place': 'Home'
    }
    recommended_titles = recommend_activity(user_input, kmeans_model, aspect_labels, instructions, tfidf_vectorizer, data)
    print("Recommended activities based on user input:")
    for title in recommended_titles:
        print(title)
        # print(instructions)  # Print the full activity text


if __name__ == "__main__":
    main()

