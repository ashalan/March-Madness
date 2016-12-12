from flask import Flask, render_template, request, jsonify, send_from_directory
import os

app = Flask(__name__)

#-------- MODEL GOES HERE -----------#
import pandas as pd
import pickle

# load the model from disk
input_features = pickle.load(open('pkl/input_features.pkl', 'rb'))
n = pickle.load(open('pkl/n.pkl', 'rb'))
outputs = pickle.load(open('pkl/outputs.pkl', 'rb'))
train_outputs = pickle.load(open('pkl/train_outputs.pkl', 'rb'))
inputs = pickle.load(open('pkl/inputs.pkl', 'rb'))
normalizer = pickle.load(open('pkl/normalizer.pkl', 'rb'))
train_inputs = pickle.load(open('pkl/train_inputs.pkl', 'rb'))
games_rnd = pickle.load(open('pkl/games_rnd.pkl', 'rb'))

def neural_result(input):
    """Call the neural network, and translates its output to a match result."""
    n_output = n.activate(input) 
    if n_output >= 0.5:
        return 2
    else:
        return 1

def output(year, team1, team2):
    inputs = []
    
    for feature in input_features:
        from_team_2 = '2' in feature
        if feature in [x for x in games_rnd.columns.values if x != 'Season']:
            if from_team_2:
                try:
                    value = games_rnd[games_rnd['team2'] == team2][games_rnd['Season'] == year].iloc[[0]][feature]
                except:
                    value = games_rnd[games_rnd['team2'] == team2].iloc[[-1]][feature]
            else:
                try:
                    value = games_rnd[games_rnd['team1'] == team1][games_rnd['Season'] == year].iloc[[0]][feature]
                except:
                    value = games_rnd[games_rnd['team1'] == team1].iloc[[-1]][feature]
        elif feature == 'Season':
            value = year
        else:
            raise ValueError("Don't know where to get feature: " + feature)
        inputs.append(value)
        
    inputs = normalizer.transform(inputs)
    result = neural_result(inputs)
    
    if result == 1:
        results = team1
    elif result == 2:
        results = team2
    else:
        results = 'Unknown result: ' + str(result)
    return results

#-------- ROUTES GO HERE -----------#
@app.route('/prediction')
def prediction():
    year = request.args.get('year', 0, type=int)
    team1 = request.args.get('team1', 0, type=str)
    team2 = request.args.get('team2', 0, type=str)
    return jsonify(result = output(year, team1, team2))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/json/<path:path>')
def send_json(path):
    return send_from_directory('json', path)

if __name__ == '__main__':
    '''Connects to the server'''
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# 74, 49, 0, 23, 7, 25