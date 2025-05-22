import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 84
MAX_SEQ_LENGTH = 24


def load_video(path, resize=(IMG_SIZE, IMG_SIZE), max_frames=MAX_SEQ_LENGTH):
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(length / max_frames))
    counter = 0
    frames = []

    try:
        while True:
            counter += 1
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)  # Resize to 84x84
            frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
            if counter % step == 0:
                frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()

    frames = np.array(frames, dtype=np.float32)
    # frames /= 255.0
    return frames


def predict_violence(model, video_path):
    frames = load_video(video_path)

    print(f"Frames shape: {frames.shape}")

    input_tensor = np.expand_dims(frames, axis=0)
    print(f"Input tensor shape: {input_tensor.shape}")

    prediction = model.predict(input_tensor)

    print(f"Prediction: {prediction}")

    violence_score = float(prediction[0][0])  # If this is between 0 and 1, you can scale it
    violence_score_percentage = round(violence_score, 2)  # Round percentage
    return violence_score_percentage


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model
