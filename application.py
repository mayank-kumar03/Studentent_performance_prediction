from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load the pre-trained model and scaler
import pickle
model = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/')
def index():
      return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
      if request.method == 'POST':
            HS=int(request.form.get('Hours Studied'))
            PS=int(request.form.get('Previous Scores'))
            EA=int(request.form.get('Extracurricular Activities'))
            SH=int(request.form.get('Sleep Hours'))
            SQ=int(request.form.get('Sample Question Papers Practiced'))

            new_data_scaled=scaler.transform([[HS, PS, EA, SH, SQ]])

            result=model.predict(new_data_scaled)

            return render_template('predict.html', results=result[0])
      else:
            return render_template('predict.html')

if __name__ == '__main__':
      app.run(debug=True, host='0.0.0.0', port=5000)