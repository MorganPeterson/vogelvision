#!/usr/bin/env python

import os
import sys
sys.path.insert(0, os.path.dirname('../vgg16'))

from flask import Flask, request
from flask_cors import CORS

from vgg16.helpers import init_vgg16_with_model, img_labels
from upload_handler import upload_handler

from constants import *

app = Flask(__name__)
CORS(app)

vgg16 = init_vgg16_with_model(MODEL_PATH, NUM_CLASSES, TORCH_DEVICE)
labels = img_labels(LABEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        return upload_handler(request, UPLOAD_DIR, TORCH_DEVICE).predict(vgg16, labels)
    return None

if __name__ == '__main__':
    app.run(debug=True)
