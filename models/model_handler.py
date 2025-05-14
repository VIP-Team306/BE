import tempfile
import tensorflow as tf


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def predict_violence(model, file_obj):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(file_obj.read())
        tmp.flush()

        prediction = model.predict(file_obj)
        violence_percentage = float(prediction[0][0])

        return violence_percentage
