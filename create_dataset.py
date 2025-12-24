import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# -----------------------------
# Hand Landmarker Task model
# -----------------------------
MODEL_PATH = r'C:\MINOR PROJECT\hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)
DATA_DIR = './data'
data = []
labels = []
for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            x_, y_ = [], []
            for lm in hand_landmarks:
                x_.append(lm.x)
                y_.append(lm.y)
            data_aux = []
            for lm in hand_landmarks:
                data_aux.extend([
                    lm.x - min(x_),
                    lm.y - min(y_)
                ])
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(int(label))
print("Total samples:", len(data))
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("✅ data.pickle created using hand_landmarker.task")