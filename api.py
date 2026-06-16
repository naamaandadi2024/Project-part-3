from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import re
from assets_data_prep import prepare_data
from transformers import ActorExperienceTransformer

app = Flask(__name__)

# טעינת המודל
model = joblib.load('trained_model.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

EXPECTED_FIELDS = ['Language', 'startYear', 'runtimeMinutes', 'genres', 'Country', 'plot', 'lead_actors_ids']

def has_no_numbers(text):
    # בודק שאין אף ספרה (0-9) בכל הטקסט
    return not bool(re.search(r'\d', str(text)))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. בדיקה שכל השדות קיימים ולא ריקים (גם אם מדובר ברווחים בלבד)
        for field in EXPECTED_FIELDS:
            if field not in data or str(data[field]).strip() == "":
                return jsonify({'error': f'Missing or empty field: {field}'}), 400

        # . בדיקות תקינות: אסור שיהיו מספרים ב-Language, Genres, Country
   
        text_fields = ['Language', 'genres', 'Country']
        for field in text_fields:
            if not has_no_numbers(data[field]):
                return jsonify({'error': f'{field} cannot contain numbers'}), 400

       # --- בדיקה : מפרקים לפי פסיק ובודקים שאין רווחים בתוך שום חלק ---
        parts = data['lead_actors_ids'].split(',')
        for part in parts:
            # אם אחרי הורדת רווחים מהקצוות עדיין נשאר רווח בפנים, זה אומר שיש רווח לא תקין
            if ' ' in part.strip():
                 return jsonify({'error': 'Invalid format: separate actors with commas only (no spaces inside the actor IDs)'}), 400
        
        
        # . בדיקת פורמט שנה (4 ספרות בדיוק)
        if not re.match(r'^\d{4}$', str(data['startYear'])):
            return jsonify({'error': 'Year must be exactly 4 digits'}), 400

        # . עיבוד הנתונים
        data['lead_actors_ids'] = str([a.strip() for a in data['lead_actors_ids'].split(',')])
        df = pd.DataFrame([data])
        df['startYear'] = df['startYear'].astype(float)
        df['runtimeMinutes'] = df['runtimeMinutes'].astype(float)
        
        processed_df = prepare_data(df)
        prediction = model.predict(processed_df)
        
        return jsonify({'predicted_rating': float(prediction[0])})

    except ValueError:
        return jsonify({'error': 'Invalid data format (check your numbers/years)'}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)