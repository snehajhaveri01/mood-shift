from flask import Flask, request, jsonify

from src.app import predict_and_suggest_activities, sentiments

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/')
def index():
    return 'Welcome to Mood Shift!'


@app.route('/get_activities', methods=['POST'])
def api_get_activities():
    data = request.get_json()
    if not data or any(key not in data for key in ['mood', 'aspect', 'place', 'reason']):
        return jsonify({'error': 'Missing required data in request'}), 400
    try:
        activities = predict_and_suggest_activities(data['aspect'], data['mood'], data['place'], data['reason'],
                                                    sentiments)
        if isinstance(activities, list):
            return jsonify({'activities': activities})
        else:
            return jsonify({'message': activities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
