from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='mood-shift.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Preprocess input data
    mood = data['mood'].lower()
    aspect = data['aspect'].lower()
    reason = data['reason'].lower()
    place = data['place'].lower()
    input_data = preprocess_input(mood, aspect, reason, place)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted mood
    predicted_index = np.argmax(output_data)
    predicted_mood = positive_emotions[predicted_index]

    return jsonify({'predicted_mood': predicted_mood})


def preprocess_input(mood, aspect, reason, place):
    # Convert mood, aspect, and place to one-hot encoded vectors
    mood_index = categorical_features.index(mood)
    aspect_index = categorical_features.index(aspect)
    place_index = categorical_features.index(place)
    input_data = np.zeros((1, len(categorical_features) + len(activities_text)))

    # Set one-hot encoded values
    input_data[0, mood_index] = 1
    input_data[0, aspect_index] = 1
    input_data[0, place_index] = 1

    # Convert reason to text vector
    reason_vector = text_transformer.transform([reason]).toarray()

    # Combine one-hot encoded vectors and text vector
    input_data[0, len(categorical_features):] = reason_vector

    return input_data


if __name__ == '__main__':
    app.run(debug=True)
