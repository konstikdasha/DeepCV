from .Blinking import BlinkingHandler
from .HeadPosition import HeadPosition
from .Emotion import EmotionDetection
from .utils import get_mediapipe_app

import cv2
import mediapipe as mp
import numpy as np
import time


class StatusController:
    def __init__(self):
        self.normal_emotions = {"neutral"}
        self.mp_face_mesh = mp.solutions.face_mesh
        self.facemesh_model = get_mediapipe_app()
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.blink_handler = BlinkingHandler()
        self.head_handler = HeadPosition()
        self.emotion_handler = EmotionDetection()

    def process(self,
                image,
                blinking_thresholds=None,
                head_thresholds=None,
                emotion_thresholds=None):

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False
        
        # Get the result
        facemesh_results = self.facemesh_model.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        orig_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        info_image = orig_image.copy()
        head_image_info = info_image

        
            
        blink_image_info, blink_result = self.blink_handler.process(orig_image, info_image, facemesh_results, blinking_thresholds)
        emotion_image_info, emotion_result = self.emotion_handler.process(orig_image, blink_image_info, emotion_thresholds)
        head_image_info, head_result = self.head_handler.process(orig_image, emotion_image_info, facemesh_results, head_thresholds)

        result = {"mp_detection": True, **blink_result, **emotion_result, **head_result}

        if result["is sleeping"] or result["emotion"] == "No detection" or result["don't look"]:
            total = "don't work"
        elif result["want to sleep / stressed"] or result["emotion"] not in self.normal_emotions:
            total = "needs a rest"
        else:
            total = "good"

        if facemesh_results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=head_image_info,
                landmark_list=facemesh_results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec)

        else:
            result = {"mp_detection": False}
            total = "don't work"

        return head_image_info, result, total