import cv2
import mediapipe as mp
import numpy as np
import time

class HeadPosition:
    def __init__(self):
            self.state_tracker = {
                                    "start_time": time.perf_counter(),
                                    "DROWSY_TIME": 0.0,
                                    "play_alarm": False,
                                 }

    def process(self, orig_image, info_image, facemesh_results, thresholds=None):

        if thresholds is None:
            thresholds = {
                          "WAIT_TIME": 1,
                          "x_max": 10,
                          "y_max": 10,
                          "z_max": 10,
                          }

        result = dict()
            
        if facemesh_results.multi_face_landmarks:

            img_h, img_w, img_c = orig_image.shape
            face_3d = []
            face_2d = []

            landmarks = facemesh_results.multi_face_landmarks[0].landmark
            for idx, lm in enumerate(landmarks):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
        

            # See where the user's head tilting
            if y < -thresholds["y_max"]:
                text = "looking Left"
            elif y > thresholds["y_max"]:
                text = "looking Right"
            elif x < -thresholds["x_max"]:
                text = "looking Down"
            elif x > thresholds["x_max"]:
                text = "looking Up"
            else:
                text = "forward"

            if text != "forward":

                end_time = time.perf_counter()

                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    # text += ", don't see to screen!"
                    self.state_tracker["play_alarm"] = True
                else:
                    self.state_tracker["play_alarm"] = False

            else:

                end_time = time.perf_counter()
                self.state_tracker["start_time"] = end_time
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["play_alarm"] = False

            # Display the nose direction
            cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(info_image, p1, p2, (255, 0, 0), 3)

            result = {"position": text,
                        "x head": str(np.round(x, 2)),
                        "y head": str(np.round(y, 2)),
                        "z head": str(np.round(z, 2)),
                        "looking away": str(self.state_tracker["DROWSY_TIME"]),
                        "don't look": self.state_tracker["play_alarm"]}
        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["play_alarm"] = False

            result = {"head position": None,
                    "x angla": None,
                    "y angle": None,
                    "z angle": None,
                    "looking away time": str(self.state_tracker["DROWSY_TIME"]),
                    "don't look": self.state_tracker["play_alarm"]}


        
        return info_image, result