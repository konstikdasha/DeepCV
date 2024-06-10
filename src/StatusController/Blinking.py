import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):

    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Calculate Eye aspect ratio

    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    return frame



class BlinkingHandler:
    def __init__(self):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        # Used for coloring landmark points.
        # Its value depends on the current EAR value.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        # Initializing Mediapipe FaceMesh solution pipeline


        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "COLOR": self.GREEN,
            "play_alarm": False,
            "blink_time": 0,
            "num_fast_blinks": 0,
            "wanna_sleep": False,
        }

        self.EAR_txt_pos = (10, 30)

    def process(self, frame: np.array, info_frame, facemesh_results, thresholds: dict=None):
        """
        This function is used to implement our Drowsy detection algorithm

        Args:
            frame: (np.array) Input frame matrix.
            thresholds: (dict) Contains the two threshold values
                               WAIT_TIME and EAR_THRESH.

        Returns:
            The processed frame and a boolean flag to
            indicate if the alarm should be played or not.
        """
        if thresholds is None:
            thresholds = {
                          "EAR_THRESH": 0.13,
                          "WAIT_TIME": 1.5,
                          "min_blink_time": 20,
                          "max_fast_blinks": 1,
                          }

        # To improve performance,
        # mark the frame as not writeable to pass by reference.
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))
        SLEEP_txt_pos = (10, int(frame_h // 2 * 1.95))
        EAR = None

        

        if facemesh_results.multi_face_landmarks:
            landmarks = facemesh_results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
            info_frame = plot_eye_landmarks(info_frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])


            if EAR < thresholds["EAR_THRESH"]:

                # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                # and reset the start_time for the next iteration.


                end_time = time.perf_counter()

                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True

            else:

                end_time = time.perf_counter()
                blink_time = end_time - self.state_tracker["start_time"]

                if self.state_tracker["DROWSY_TIME"] != 0:

                    if self.state_tracker["blink_time"] < thresholds["min_blink_time"]:

                        self.state_tracker["num_fast_blinks"] += 1
                        if self.state_tracker["num_fast_blinks"] >= thresholds["max_fast_blinks"]:
                            self.state_tracker["wanna_sleep"] = True

                    else:
                        self.state_tracker["wanna_sleep"] = False
                        self.state_tracker["num_fast_blinks"] = 0


                if self.state_tracker["DROWSY_TIME"] == 0:
                    self.state_tracker["blink_time"] += end_time - self.state_tracker["start_time"]
                else:
                    self.state_tracker["blink_time"] = 0



                self.state_tracker["start_time"] = end_time
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            EAR_txt = f"EAR: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"


        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False

        result = {"want to sleep / stressed": self.state_tracker["wanna_sleep"],
                  "num fast blinks": self.state_tracker["num_fast_blinks"],
                  "is sleeping": self.state_tracker["play_alarm"],
                  "drowsy time": self.state_tracker["DROWSY_TIME"],
                  "blink time": self.state_tracker["blink_time"],
                  "EAR": EAR,
                  }


        return info_frame, result
