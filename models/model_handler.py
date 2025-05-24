import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

IMG_SIZE = 84
MAX_SEQ_LENGTH = 24

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)


def get_caption(image: Image.Image, prompt: str) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,  # Enable sampling for varied outputs
            top_p=0.9,  # Nucleus sampling parameter
            temperature=0.7,  # Temperature controls randomness
            num_return_sequences=1,
        )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def describe_violence_with_blip(video_path: str, load_video_fn, violence_score: float, max_frames: int = 1):
    frames = load_video_fn(video_path)
    if len(frames) == 0:
        return ["No frames extracted from the video."]

    step = max(1, len(frames) // max_frames)
    selected_frames = frames[::step][:max_frames]

    descriptions = []
    for idx, frame in enumerate(selected_frames):
        # Convert float32 frame to uint8 image and resize to BLIP-2 expected size (224x224)
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        image = Image.fromarray(frame_uint8).resize((224, 224))

        # Create prompt per frame to encourage variation
        prompt = f"Frame {idx + 1}: Describe any violent or aggressive actions in this scene."
        description = get_caption(image, prompt)
        descriptions.append(f"Frame {idx + 1}: {description}")

    return descriptions


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

    violence_score = float(prediction[0][0])
    print(f"Violence score: {violence_score * 100}%")
    violence_score_percentage = round(violence_score, 2)  # Round to 2 decimal places
    print(f"Rounded violence score: {violence_score_percentage}")

    if violence_score_percentage > 0.5:
        descriptions = describe_violence_with_blip(video_path, load_video, violence_score_percentage)
        for desc in descriptions:
            print(desc)

    return violence_score_percentage


def load_model(model_path):
    rgb_model = tf.keras.models.load_model(model_path, compile=False)
    return rgb_model
