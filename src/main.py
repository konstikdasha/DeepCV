import cv2
from StatusController import StatusController
from tkinter import *
from GUI import GUI
import time as t

if __name__ == '__main__':

    status_change_check = ['','']
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('./data/test2.mp4')
    controller = StatusController()
    root = Tk()
    root.geometry('1800x800')
    root.state('zoomed')
    GUI_obj = GUI(root)
    flags = [True,True]
    callback_dict = {}
    blinking_thresholds = {
        "EAR_THRESH": 0.13,
        "WAIT_TIME": 1.5,
        "min_blink_time": 20,
        "max_fast_blinks": 1,
    }
    head_thresholds = {
        "WAIT_TIME": 1,
        "x_max": 10,
        "y_max": 10,
        "z_max": 10,
    }

    while cap.isOpened():
        start_time = t.time()
        success, image = cap.read()

        image_info, state_dict, status = controller.process(image, blinking_thresholds, head_thresholds)
        status_change_check[1] = status
        flags[1] = True if (status_change_check[0] != status_change_check[1]) else False
        if state_dict['mp_detection'] == True:
            state_dict['x head'] = float(state_dict['x head'])
            state_dict['y head'] = float(state_dict['y head'])
            state_dict['z head'] = float(state_dict['z head'])
            state_dict['looking away'] = float(state_dict['looking away'])
            callback_dict = GUI_obj.running_loop(image_info, state_dict, status, flags[0], (1 / (t.time() - start_time)), flags[1])
            blinking_thresholds = {
                "EAR_THRESH": callback_dict[6].get(),
                "WAIT_TIME": callback_dict[4].get(),
                "min_blink_time": callback_dict[5].get(),
                "max_fast_blinks": callback_dict[2].get(),
            }
            head_thresholds = {
                "WAIT_TIME": callback_dict[12].get(),
                "x_max": callback_dict[9].get(),
                "y_max": callback_dict[10].get(),
                "z_max": callback_dict[11].get()
            }
        else:
            GUI_obj.running_loop(image_info, state_dict,status, flags[0], (1/(t.time()-start_time)), flags[1])
            blinking_thresholds = {
                "EAR_THRESH": 0.13,
                "WAIT_TIME": 1.5,
                "min_blink_time": 20,
                "max_fast_blinks": 1,
            }
            thresholds = {
                "WAIT_TIME": 1,
                "x_max": 10,
                "y_max": 10,
                "z_max": 10,
            }
        flags[0] = False


    cap.release()