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


def describe_violence_with_blip(video_path: str, load_video_fn, max_frames: int = 1):
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


def load_video_segments(video_path, segment_seconds=8, stride_seconds=4, target_frames=24, resize=(84, 84)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_per_segment = int(fps * segment_seconds)
    stride_frames = int(fps * stride_seconds)

    segments = []
    timestamps = []

    if total_frames < frames_per_segment:
        # Short video fallback
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        segment = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            segment.append(frame)

        if segment:
            if len(segment) >= target_frames:
                segment = segment[:target_frames]
            else:
                segment += [segment[-1]] * (target_frames - len(segment))  # pad

            segments.append(np.array(segment, dtype=np.float32))
            timestamps.append((0.0, round(total_frames / fps, 2)))

    else:
        current_start = 0
        while current_start + frames_per_segment <= total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_start)
            segment = []

            for _ in range(frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                segment.append(frame)

            if not segment:
                break

            if len(segment) >= target_frames:
                segment = segment[:target_frames]
            else:
                segment += [segment[-1]] * (target_frames - len(segment))  # pad

            segments.append(np.array(segment, dtype=np.float32))
            timestamps.append((
                round(current_start / fps, 2),
                round((current_start + frames_per_segment) / fps, 2)
            ))

            current_start += stride_frames

    cap.release()
    return segments, timestamps


def predict_violence_per_segment(model, video_path, threshold=0.5):
    segments, timestamps = load_video_segments(video_path)

    for i, (segment, (start, end)) in enumerate(zip(segments, timestamps)):
        input_tensor = np.expand_dims(segment, axis=0)  # shape: (1, 24, 84, 84, 3)
        try:
            print(start, end)
            prediction = model.predict(input_tensor, verbose=0)
            violence_score = float(prediction[0][0])
            violence_score_percentage = round(violence_score, 2)

            if violence_score > threshold:
                # print(describe_violence_with_blip(video_path, load_video_segments))
                return [{
                    "segment_index": i,
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "score": violence_score_percentage
                }]
        except Exception as e:
            print(f"Error predicting segment {i}: {e}")

    # Return empty list if no violent segments found
    return []


def load_model(model_path):
    rgb_model = tf.keras.models.load_model(model_path, compile=False)
    return rgb_model
