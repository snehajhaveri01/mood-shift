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

