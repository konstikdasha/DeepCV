import cv2
import mediapipe as mp
import numpy as np
import time
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'\n\nUsing {device} device\n\n')

class EmotionDetection:
    def __init__(self):
            self.state_tracker = {
                                    "start_time": time.perf_counter(),
                                    "EMO_TIME": 0,
                                 }

            self.face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2022mar.onnx", "", (320, 320))

            # Load model directly
            self.processor = AutoImageProcessor.from_pretrained("adhityamw11/facial_emotions_image_detection_rafdb_microsoft_vit")
            self.model = AutoModelForImageClassification.from_pretrained("adhityamw11/facial_emotions_image_detection_rafdb_microsoft_vit").to(device)
            self.current_text = None
            self.previous_text = None
            self.first = True

    def process(self, orig_image, info_image, thresholds=None):

        (h, w) = orig_image.shape[:2]

        if thresholds is None:
            thresholds = {
                          "WAIT_TIME": 0.2,
                          "confidence": 0.7,
                          }

        img_W = int(img.shape[1])
        img_H = int(img.shape[0])
        # Set input size
        detector.setInputSize((img_W, img_H))

        detections = detector.detect(img)

        faces = list()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > thresholds["confidence"]:
                # Get the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))

                # Draw a rectangle around the face
                cv2.rectangle(info_image, (startX, startY), (endX, endY), (0, 0, 255), 2)

        if len(faces) > 1:
            result = 'More one'
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["EMO_TIME"] = 0.0
        elif len(faces) == 1:
            face = orig_image[faces[0][0]:faces[0][2],
                              faces[0][1]:faces[0][3]]

            inputs = self.processor(orig_image, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_label = logits.argmax(-1).item()
            result = self.model.config.id2label[predicted_label]
        else:
            result = 'No detection'
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["EMO_TIME"] = 0.0

        print(result)
        print(self.previous_text)

        if self.first:
            self.first = False
            self.current_text = result
            self.previous_text = result

        elif result == self.previous_text:
            self.state_tracker['EMO_TIME'] += time.perf_counter() - self.state_tracker["start_time"]
            self.state_tracker["start_time"] = time.perf_counter()
        else:
            self.previous_text = result
            result = self.current_text
            self.state_tracker['EMO_TIME'] = 0
            self.state_tracker["start_time"] = time.perf_counter()

        if self.state_tracker['EMO_TIME'] > thresholds['WAIT_TIME']:
            self.previous_text = result
            self.current_text = result

        result = {"emotion": self.current_text}
        
        return info_image, result
