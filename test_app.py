"""Pytest suite for the digit-recognition Flask app."""
# pylint: disable=missing-function-docstring
import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import app as app_module
import model


@pytest.fixture(name="client")
def client_fixture():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def make_png(size=(50, 50)):
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def post_file(client, fileobj, name):
    return client.post("/prediction", data={"file": (fileobj, name)},
                       content_type="multipart/form-data")


def test_home_page(client):
    assert client.get("/").status_code == 200


def test_valid_upload(client):
    with patch.object(app_module, "predict_result", return_value=7) as mock_pred:
        response = post_file(client, make_png(), "hand.png")
    assert response.status_code == 200
    mock_pred.assert_called_once()


def test_missing_file(client):
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_invalid_file_content(client):
    response = post_file(client, io.BytesIO(b"not an image"), "bad.txt")
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_get_not_allowed_on_prediction(client):
    assert client.get("/prediction").status_code == 405


def test_preprocess_shape_and_range():
    arr = model.preprocess_img(make_png((300, 120)))
    assert arr.shape == (1, 224, 224, 3)
    assert 0.0 <= arr.min() <= arr.max() <= 1.0


def test_predict_result_returns_argmax():
    with patch.object(model, "get_model") as mock_get:
        mock_get.return_value.predict.return_value = np.array([[0.1, 0.05, 0.8, 0.05]])
        assert model.predict_result(np.zeros((1, 224, 224, 3))) == 2
