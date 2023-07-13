from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model from the saved pickle file
with open('model_cancer.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json()

    # Extract the features from the request data
    features = [data['mean_radius'], data['mean_texture'], data['mean_perimeter'],
                data['mean_area'], data['mean_smoothness'], data['mean_compactness'],
                data['mean_concavity'], data['mean_concave_points'], data['mean_symmetry'],
                data['mean_fractal_dimension'], data['radius_error'], data['texture_error'],
                data['perimeter_error'], data['area_error'], data['smoothness_error'],
                data['compactness_error'], data['concavity_error'], data['concave_points_error'],
                data['symmetry_error'], data['fractal_dimension_error'], data['worst_radius'],
                data['worst_texture'], data['worst_perimeter'], data['worst_area'], data['worst_smoothness'],
                data['worst_compactness'], data['worst_concavity'], data['worst_concave_points'],
                data['worst_symmetry'], data['worst_fractal_dimension']]

    # Make the prediction using the loaded model
    prediction = model.predict([features])[0]

    # Return the prediction as a response
    response = {'prediction': int(prediction)}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)