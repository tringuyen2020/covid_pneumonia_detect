import os
import time
import keras
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

# Load the model once when the server starts
RESNET50_MODEL_PATH = os.path.join(os.path.dirname((__file__)), 'models', 'covid_pneumonia_resnet50_model.keras')
VGG16_MODEL_PATH = os.path.join(os.path.dirname((__file__)), 'models', 'covid_pneumonia_vgg16_model.keras')
XCEPTION_MODEL_PATH = os.path.join(os.path.dirname((__file__)), 'models', 'covid_pneumonia_xception_model.keras')
resnet50_model = load_model(RESNET50_MODEL_PATH)
vgg16_model = load_model(VGG16_MODEL_PATH)
xception_model = load_model(XCEPTION_MODEL_PATH)

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
            resnet_start = time.time()
            resnet_predictions = resnet50_model.predict(img_array)
            print(f'ResNet50 Predictions: {resnet_predictions}')
            resnet_predicted_idx = np.argmax(resnet_predictions[0])
            resnet_confidence = resnet_predictions[0][resnet_predicted_idx]*100
            resnet_end = time.time()
            resnet_inference_time = resnet_end - resnet_start

            vgg_start = time.time()
            vgg_predictions = vgg16_model.predict(img_array)
            print(f'VGG16 Predictions: {vgg_predictions}')
            vgg_predicted_idx = np.argmax(vgg_predictions[0])
            vgg_confidence = vgg_predictions[0][vgg_predicted_idx]*100
            vgg_end = time.time()
            vgg_inference_time = vgg_end - vgg_start

            xception_start = time.time()
            xception_predictions = xception_model.predict(img_array)
            print(f'Xception Predictions: {xception_predictions}')
            xception_predicted_idx = np.argmax(xception_predictions[0])
            xception_confidence = xception_predictions[0][xception_predicted_idx]*100
            xception_end = time.time()
            xception_inference_time = xception_end - xception_start

            result = {
                'resnet_predicted_class': CLASS_LABELS[resnet_predicted_idx],
                'resnet_confidence': f"{resnet_confidence:.2f}%",
                'resnet_inference_time': f"{resnet_inference_time:.4f} seconds",
                'vgg_predicted_class': CLASS_LABELS[vgg_predicted_idx],
                'vgg_confidence': f"{vgg_confidence:.2f}%",
                'vgg_inference_time': f"{vgg_inference_time:.4f} seconds",
                'xception_predicted_class': CLASS_LABELS[xception_predicted_idx],
                'xception_confidence': f"{xception_confidence:.2f}%",
                'xception_inference_time': f"{xception_inference_time:.4f} seconds",
                'image_url': image_url
            }
        except Exception as e:
            result = {'error': f"An error occurred: {str(e)}"}
            # Delete the temporary file
            if os.path.exists(image_path):
                os.remove(image_path)

        return render(request, 'predictor/index.html', {'result': result})
    return render(request, 'predictor/index.html')