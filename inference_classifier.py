import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
model = pickle.load(open('model.p', 'rb'))['model']
cap = cv2.VideoCapture(0)
MODEL_PATH = r'C:\MINOR PROJECT\hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)
labels_dict = {0: 'A', 1: 'B', 2: 'C'}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, frame_rgb)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        x_, y_ = [], []
        data_aux = []
        for lm in hand_landmarks:
            x_.append(lm.x)
            y_.append(lm.y)
        for lm in hand_landmarks:
            data_aux.extend([
                lm.x - min(x_),
                lm.y - min(y_)
            ])
        if len(data_aux) == 42:
            pred = model.predict([np.asarray(data_aux)])
            char = labels_dict[int(pred[0])]
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                char,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
    cv2.imshow("Sign Language Detection (A B C)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
