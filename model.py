"""Image preprocessing and digit prediction helpers."""
from functools import lru_cache

import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

MODEL_PATH = "digit_model.h5"


@lru_cache(maxsize=1)
def get_model():
    """Load the trained Keras model once and cache it."""
    return load_model(MODEL_PATH)


def preprocess_img(img_path):
    """Resize an image to 224x224, scale to [0, 1], return shape (1, 224, 224, 3)."""
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    return img2arr.reshape(1, 224, 224, 3)


def predict_result(predict):
    """Return the predicted digit for a preprocessed image array."""
    pred = get_model().predict(predict)
    return np.argmax(pred[0], axis=-1)
