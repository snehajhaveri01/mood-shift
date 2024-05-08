from flask import Flask, request, jsonify
from flask_cors import CORS

from src.app import predict_and_suggest_activities

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/')
def index():
    return 'Welcome to Mood Shift!'


@app.route("/test", methods=["POST"])
def test():
    data = request.get_json()
    return jsonify(data)


@app.route('/activities', methods=['POST'])
def activities():
    data = request.get_json()
    print(data)
    if not data or any(key not in data for key in ['mood', 'aspect', 'place', 'reason']):
        return jsonify({'error': 'Missing required data in request'}), 400
    try:
        suggested_activities = predict_and_suggest_activities(data['aspect'], data['mood'], data['place'],
                                                              data['reason'])
        if isinstance(suggested_activities, list):
            return jsonify({'activities': suggested_activities})
        else:
            return jsonify({'message': suggested_activities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
