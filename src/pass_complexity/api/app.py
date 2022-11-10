import re
import logging
import tensorflow as tf
from keras.utils import pad_sequences
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)

logger = logging.getLogger('waitress')

model = tf.keras.models.load_model('models/model.pb')

with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


@app.route('/predict', methods=['GET'])
def get_prediction():
    password = request.args.get('password')
    if password is None:
        return 'Arg password is required', 400
    password = re.findall('\S', str(password))
    tokenized_password = tokenizer.texts_to_sequences([password])
    tokenized_password = pad_sequences(tokenized_password, 83, padding='post')
    prediction = model.predict(tokenized_password)
    return jsonify(prediction=float(prediction[0][0]))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
