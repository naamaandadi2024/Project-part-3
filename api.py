from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from assets_data_prep import prepare_data
from transformers import ActorExperienceTransformer

app = Flask(__name__)

model = joblib.load('trained_model.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

EXPECTED_FIELDS = ['Language', 'startYear', 'runtimeMinutes', 'genres', 'Country', 'plot', 'lead_actors_ids']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    for field in EXPECTED_FIELDS:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    try:
        df = pd.DataFrame([data])
        df['startYear'] = df['startYear'].astype(float)
        df['runtimeMinutes'] = df['runtimeMinutes'].astype(float)
        processed_df = prepare_data(df)
        prediction = model.predict(processed_df)
        return jsonify({'predicted_rating': float(prediction[0])})

    except ValueError:
        return jsonify({'error': 'Invalid data format'}), 400
    except Exception:
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)