import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from collections import deque
import mediapipe as mp

def is_index_up(lm):
    # Index finger up
    index_up = lm[8].y < lm[6].y

    # Other fingers down
    middle_down = lm[12].y > lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y

    return index_up and middle_down and ring_down and pinky_down

# -------- MEDIAPIPE --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# -------- MODEL --------
MODEL_PATH = "gesture_model_3class.pth"
IMG_SIZE = 224
BUFFER_SIZE = 5# FIX 3: increased from 12 to 25 for better smoothing (~0.8s at 30fps)

THRESHOLDS = {
    'c': 0.6,       # reverted to original (model confidence too low at 0.75)
    'fist': 0.85,   # kept raised
    'index': 0.4,  # kept raised  # reverted to original (model confidence too low at 0.75)
}

device = torch.device("cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

class_names = ['c', 'fist', 'index']
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------- VARIABLES --------
pred_buffer = deque(maxlen=BUFFER_SIZE)

canvas = None
prev_x, prev_y = None, None
mode = "idle"
clear_counter = 0
draw_hold = 0
stable_gesture = "none"
stable_count = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    h, w, _ = frame.shape

    box_size = 350

    # -------- MEDIAPIPE (run before DL so we can gate on hand presence + drive ROI) --------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # FIX 2: only trust DL gesture when MediaPipe actually sees a hand
    hand_detected = result.multi_hand_landmarks is not None

    # -------- ROI (follows palm center if hand detected, else falls back to screen center) --------
    # palm center = average of landmarks 0 (wrist), 5, 9, 13, 17 (knuckle bases)
    if hand_detected:
        lm = result.multi_hand_landmarks[0].landmark
        palm_indices = [0, 5, 9, 13, 17]
        roi_cx = int(np.mean([lm[i].x for i in palm_indices]) * w)
        roi_cy = int(np.mean([lm[i].y for i in palm_indices]) * h) - 50
    else:
        roi_cx = w // 2
        roi_cy = h // 2

    x1 = max(0, roi_cx - box_size // 2)
    y1 = max(0, roi_cy - box_size // 2)
    x2 = min(w, x1 + box_size)
    y2 = min(h, y1 + box_size)

    roi = frame[y1:y2, x1:x2]
    cv2.imshow("ROI", roi)
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    img = transform(roi_resized).unsqueeze(0)

    # -------- DL --------
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

        pred_class = class_names[predicted.item()]
        conf_val = confidence.item()

        print(f"Pred: {pred_class}, Conf: {conf_val:.2f}")

        threshold = THRESHOLDS.get(pred_class, 0.5)
        if hand_detected:
            lm = result.multi_hand_landmarks[0].landmark

            if is_index_up(lm):
                gesture = "index"
            else:
                gesture = pred_class
        else:
            gesture = "none"
    # gate: override to "none" if MediaPipe sees no hand
    if not hand_detected:
        gesture = "none"

    pred_buffer.append(gesture)

    # count occurrences
    counts = {g: pred_buffer.count(g) for g in set(pred_buffer)}
    top_gesture = max(counts, key=counts.get)

    # stability check
    if top_gesture == stable_gesture:
        stable_count += 1
    else:
        stable_gesture = top_gesture
        stable_count = 1

    # only accept if stable for few frames
    if stable_count >= 2:
        final_gesture = stable_gesture
    else:
        final_gesture = "none"


# -------- MODE (IMPROVED WITH BUFFER + STICKINESS) --------

    if final_gesture == "index":
        mode = "draw"
        draw_hold = 10
        clear_counter = 0

    elif draw_hold > 0:
        mode = "draw"
        draw_hold -= 1

    elif final_gesture == "fist":
        mode = "idle"
        prev_x, prev_y = None, None
        draw_hold = 0   # 🔥 IMPORTANT: force stop drawing
        clear_counter = 0
        mode = "idle"
        prev_x, prev_y = None, None
        draw_hold = 0
        clear_counter = 0

    elif final_gesture == "c":
        mode = "idle"
        clear_counter += 1

        if clear_counter > 10:
            canvas = np.zeros_like(frame)
            clear_counter = 0

    else:
        mode = "idle"
        prev_x, prev_y = None, None
        draw_hold = 0
        clear_counter = 0
    # -------- MEDIAPIPE TRACKING --------
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            cx = int(lm[8].x * w)
            cy = int(lm[8].y * h)

            # -------- DRAW --------
            if mode == "draw":
                if prev_x is not None:
                    dist = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)

                    if dist < 80:
                        cx = int(0.7 * prev_x + 0.3 * cx)   
                        cy = int(0.7 * prev_y + 0.3 * cy)

                        cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255,255,255), 6)

                prev_x, prev_y = cx, cy

            else:
                prev_x, prev_y = None, None

            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    # -------- DISPLAY --------
    combined = cv2.add(frame, canvas)

    cv2.rectangle(combined, (x1, y1), (x2, y2), (0,255,0), 2)
    # -------- CLEAN DISPLAY LABEL --------
    if final_gesture == "index":
        display_mode = "DRAW"
    elif final_gesture == "fist":
        display_mode = "IDLE"
    elif final_gesture == "c":
        display_mode = "CLEAR"
    else:
        display_mode = "IDLE"

    cv2.putText(combined, f"Mode: {display_mode}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("FINAL HYBRID", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()