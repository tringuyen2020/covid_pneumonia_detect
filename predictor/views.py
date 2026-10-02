import os
import keras
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

# Load the model once when the server starts
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'covid_pneumonia_resnet50_model.keras')
model = load_model(MODEL_PATH)

# Class mappings matching the generator indices
CLASS_LABELS = ['COVID19', 'NORMAL', 'PNEUMONIA']

# Create your views here.
def index(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        image_url = fs.url(filename)
        print(f'URL: {image_url}')
        image_path = fs.path(filename)
        print(f'Image Path: {image_path}')

        try:
            # Preprocess the image
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make prediction
            predictions = model.predict(img_array)
            print(f'Predictions: {predictions}')
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]*100

            result = {
                'predicted_class': CLASS_LABELS[predicted_idx],
                'confidence': f"{confidence:.2f}%",
                'image_url': image_url
            }
        except Exception as e:
            result = {'error': f"An error occurred: {str(e)}"}
            # Delete the temporary file
            if os.path.exists(image_path):
                os.remove(image_path)

        return render(request, 'predictor/index.html', {'result': result})
    return render(request, 'predictor/index.html')