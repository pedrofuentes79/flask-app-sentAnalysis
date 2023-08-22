from flask import Flask, request, jsonify, render_template
from bert_model import SentimentClassifier

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_sentence = request.form.get('input_sentence')
    prediction = SentimentClassifier.predict_text(input_sentence)
    sentiment = "Positive" if prediction > 0.65 else "Negative"

    confidence = (1-prediction) if prediction < 0.5 else prediction

    formatted_confidence = "{:.2f}%".format(confidence*100)

    return render_template('index.html', input_sentence=input_sentence, sentiment=sentiment, confidence=formatted_confidence)


@app.route('/predict/api', methods=['POST'])
def predict_api():
    data = request.get_json()
    input_sentence = data['input_sentence']
    
    prediction = SentimentClassifier.predict_text(input_sentence)  

    return jsonify({"input_sentence": input_sentence, "prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)