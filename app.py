from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import numpy as np
import tensorflow as tf
from PIL import Image
from my_ml_project import get_animals_detection_images_dataset, data_preprocessing, build_model, train_model, evaluate_model

app = Flask(__name__)
api = Api(app)

# Load the trained model
model = build_model()
train_images, train_labels, test_images, test_labels = data_preprocessing(get_animals_detection_images_dataset())
train_model(model, train_images, train_labels)
test_loss, test_acc = evaluate_model(model, test_images, test_labels)

class AnimalDetector(Resource):
    def post(self):
        # Receive an image file from the client
        try:
            image_file = request.files['image']
            # Ensure that the image file is a valid format
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            if '.' not in image_file.filename or image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return jsonify({'error': 'Invalid image format. Please upload a png or jpg/jpeg file.'}), 400
            # Ensure that the image file is not too large
            max_size = 2 * 1024 * 1024 # 2MB
            if len(image_file.read()) > max_size:
                return jsonify({'error': 'Image file too large. Maximum file size is 2MB.'}), 400
            image_file.seek(0) # Reset file pointer
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        
        # Convert the image to a numpy array
        try:
            image = np.array(Image.open(image_file))
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        
        # Preprocess the image (resize, normalize, etc.)
        processed_image = preprocess_image(image)
        
        # Make a prediction using the model
        if model is None:
            return jsonify({'error': 'Model not loaded.'}), 500
        try:
            prediction = model.predict(processed_image)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        # Convert the prediction to a string (e.g. "dog")
        class_names = ['dog', 'cat', 'zebra', 'lion', 'leopard', 'cheetah', 'tiger', 'bear', 'brown bear', 'butterfly', 'canary', 'crocodile', 'polar bear', 'bull', 'camel', 'crab', 'chicken', 'centipede', 'cattle', 'caterpillar', 'duck']
        predicted_class = class_names[np.argmax(prediction)]
        
        # Return the prediction to the client
        return jsonify({'prediction': predicted_class})

api.add_resource(AnimalDetector, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
