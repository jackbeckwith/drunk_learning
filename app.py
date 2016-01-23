from flask import Flask, request, g
from drunk_learning import DrunkLearningNB, DrunkLearningSVM
app = Flask(__name__)
DrunkLearning = DrunkLearningNB()

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def send_prediction():
    data = dict(request.get_json())
    X = data.get('X')
    return DrunkLearning.predict(X)

@app.route('/log', methods=['POST'])
def log_data():
    # Put data into db
    data = dict(request.get_json())
    print data
    return 'OK'

@app.route('/fit', methods=['POST'])
def fit():
    data = dict(request.get_json())
    X = data.get('X')
    y = data.get('y')
    DrunkLearning.fit(X, y)
    return 'OK'

@app.route('/partial_fit', methods=['POST'])
def partial_fit():
    data = dict(request.get_json())
    X = data.get('X')
    y = data.get('y')
    DrunkLearning.partial_fit(X, y)
    return 'OK'

if __name__ == '__main__':
    app.debug = True
    app.run()