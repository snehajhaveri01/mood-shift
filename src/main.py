from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import json

app = Flask(__name__)

# Load the data and preprocess it
def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    aspect_labels = []
    instructions = []
    for item in data:
        aspect_labels.extend([item['aspect']] * len(item['instructions']))
        instructions.extend(item['instructions'])
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(instructions)
    
    kmeans_model = KMeans(n_clusters=3)
    kmeans_model.fit(tfidf_matrix)
    
    return data, aspect_labels, instructions, tfidf_vectorizer, kmeans_model

# Recommend activities based on user input
def recommend_activity(user_input, data, aspect_labels, instructions, tfidf_vectorizer, kmeans_model):
    aspect = user_input['Aspect']
    aspect_indices = [i for i, label in enumerate(aspect_labels) if label.lower() == aspect.lower()]
    
    if len(aspect_indices) == 6:
        recommended_indices = aspect_indices
    else:
        aspect_tfidf_matrix = tfidf_vectorizer.transform([instructions[i] for i in aspect_indices])
        predicted_cluster = kmeans_model.predict(aspect_tfidf_matrix)
        common_cluster = np.bincount(predicted_cluster).argmax()
        recommended_indices = np.where(predicted_cluster == common_cluster)[0]
        recommended_indices = recommended_indices[:6]

    recommended_titles = [data[i]['title'] for i in recommended_indices]
    
    return recommended_titles 

# Load and preprocess data
file_path = 'activities.json'
activities_data, aspect_labels, instructions, tfidf_vectorizer, kmeans_model = load_and_preprocess_data(file_path)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    user_input = request.json
    recommended_titles = recommend_activity(user_input, activities_data, aspect_labels, instructions, tfidf_vectorizer, kmeans_model)
    return jsonify({"Recommended Activities": recommended_titles})

if __name__ == '__main__':
    app.run(debug=True)
