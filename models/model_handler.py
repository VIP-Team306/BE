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
    # Preprocess image + prompt into tensors for the model
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # Disable gradient calculation (inference only, no training)
    with torch.no_grad():
        # Generate caption using sampling for more varied text
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,  # Limit caption length
            do_sample=True,  # Enable random sampling instead of greedy decoding
            top_p=0.9,  # Use nucleus sampling (keep top 90% prob mass)
            temperature=0.7,  # Control randomness (lower = safer, higher = more random)
            num_return_sequences=1,  # Only generate one caption
        )

    # Decode tokens into text, remove special tokens, and strip whitespace
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def describe_violence_with_blip_from_frames(frames: list[np.ndarray], max_frames: int = 1) -> list[str]:
    # If no frames are provided, return a default message
    if len(frames) == 0:
        return ["No frames extracted."]

    # Pick evenly spaced frames up to max_frames
    step = max(1, len(frames) // max_frames)
    selected_frames = frames[::step][:max_frames]

    descriptions = []
    for idx, frame in enumerate(selected_frames):
        # Convert frame to [0–255] uint8 image format
        frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

        # Convert numpy array to PIL image and resize to model input size
        image = Image.fromarray(frame_uint8).resize((224, 224))

        # Custom prompt for each frame
        prompt = f"Frame {idx + 1}: Describe any violent or aggressive actions in this scene."

        # Get caption from BLIP model
        description = get_caption(image, prompt)

        # Save frame number + description
        descriptions.append(f"Frame {idx + 1}: {description}")

    return descriptions


def extract_frames_from_segment(video_path: str, start_time: float, end_time: float, resize=(84, 84)) -> list[
    np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return frames


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
    most_violence_video = {"score": 0}
    for i, (segment, (start, end)) in enumerate(zip(segments, timestamps)):
        input_tensor = np.expand_dims(segment, axis=0)  # shape: (1, 24, 84, 84, 3)
        try:
            # print(start, end)
            prediction = model.predict(input_tensor, verbose=0)
            violence_score = float(prediction[0][0])

            if violence_score > threshold:
                violent_frames = extract_frames_from_segment(video_path, start, end)
                descriptions = describe_violence_with_blip_from_frames(violent_frames)
                # for line in descriptions:
                #     print(line)

                return [{
                    "segment_index": i,
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "description": descriptions[0].removeprefix("Frame 1: "),
                    "score": round(violence_score, 2)
                }]

            #for no violence videos
            if most_violence_video["score"] < violence_score:
                most_violence_video = {
                    "segment_index": i,
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "description": None,
                    "score": violence_score,
                    #"violence_score": violence_score
                }

        except Exception as e:
            print(f"Error predicting segment {i}: {e}")

    # Return empty list if no violent segments found
    return [most_violence_video]


def load_model(model_path):
    rgb_model = tf.keras.models.load_model(model_path, compile=False)
    return rgb_model
