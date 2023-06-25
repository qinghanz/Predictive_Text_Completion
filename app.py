from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def hello():
    return 'Hello from the backend!'

# Load the pre-trained GPT-3 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate predicted text using the GPT-2 model
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'predictedText': predicted_text})

if __name__ == '__main__':
    print("running!")
    app.run(port=8000)