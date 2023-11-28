from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your pre-trained Keras model
keras_model = load_model('path/to/your/keras_model.h5')

# Load your scikit-learn components
scaler = StandardScaler()
scaler = scaler.fit(your_training_data)  # Update with your actual training data

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']

    # Check if the file is provided and has an allowed extension
    if file and allowed_file(file.filename):
        # Save the file to the upload folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform any necessary preprocessing on the file
        # For example, read data from the file and transform it
        # This step will depend on the specific requirements of your model
        processed_data = process_uploaded_file(file_path, scaler)

        # Make predictions using the loaded Keras model
        prediction = keras_model.predict(np.array(processed_data).reshape(1, -1))

        # Return the predictions as JSON
        return jsonify({'prediction': prediction.tolist()})

    else:
        return jsonify({'error': 'Invalid file or file format'})

def process_uploaded_file(file_path, scaler):
    # Placeholder function for preprocessing the uploaded file
    # You need to implement this based on your specific requirements
    # For example, read data from the file, transform it, and return the processed data
    # Make sure the processed data has the same format as your model expects

    # Example: Reading a CSV file
    data = np.loadtxt(file_path, delimiter=',')  # Update the delimiter based on your file format

    # Example: Scaling the data using a scaler
    scaled_data = scaler.transform(data)

    # Example: Returning the processed data
    return scaled_data.tolist()

if __name__ == '__main__':
    app.run(debug=True)
