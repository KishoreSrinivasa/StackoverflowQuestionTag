#IMPORTING THE PACKAGES REQUIRED
from flask import Flask, request, jsonify, render_template
from tagpredictor import get_Tags

# Create the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Define a route for model inference
@app.route('/predict', methods=['POST','GET'])
def predict():
    question=request.form['question']
    result = get_Tags(question)[0]
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
