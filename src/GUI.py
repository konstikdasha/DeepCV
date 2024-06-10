from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from datetime import datetime
import math as m
import influxdb_client


class GUI:
    token = "l4Ow2LAPv_8XX2uC4ay30O6rv6gnwf-YWbz3PI6s0g0CQnmhYEU3iov_6KspTaFmbx5R9wdM91ulnch4Xh_adA=="
    org = "RSU of Oil and Gas"
    url = "http://localhost:8086"
    bucket = "status_controller_bucket"
    status_label = None
    fps_label = None
    value_label = None
    image_container = None
    canvas = None
    label_list = []
    dict_state = {}
    dict_state_renamed = {}
    numerical = 'numerical'
    word = 'word'
    TF = 'T/F'
    NoData = 'NO DATA'
    dict_state_qualities = {'mp_detection': TF, 'want to sleep / stressed': TF, 'num fast blinks': numerical,
                            'is sleeping': TF, 'drowsy time': numerical, 'blink time': numerical,
                            'EAR': numerical, 'emotion': word,
                            'position': word, 'x head': numerical,
                            'y head': numerical, 'z head': numerical, 'looking away': numerical, "don't look": TF}
    f1 = None
    f1_1 = None
    f1_2 = None
    f2 = None
    f2_1 = None
    f2_2 = None
    width, hight = 0, 0
    app = None

    def __init__(self,root):
        self.client = influxdb_client.InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api()
        self.query_api = self.client.query_api()

        self.app = root
        self.app.config()

        self.width, self.height = 600, 600

        self.f1 = Frame(self.app, height=770, width=820, highlightbackground='black', highlightthickness=2, padx=50, pady=50)
        self.f1.pack(side=LEFT, padx=(15,0))
        self.f1.propagate(False)

        self.f1_1 = Frame(self.f1, height=120, width=900)
        self.f1_1.pack(side=TOP)
        self.f1_1.propagate(False)

        self.f1_2 = Frame(self.f1, height=680, width=900)
        self.f1_2.pack()
        self.f1_2.propagate(False)

        self.f2 = Frame(self.app, width=900, height=800)
        self.f2.pack(side='left')
        self.f2.propagate(False)
        for i in range(2): self.f2.rowconfigure(index=0, minsize=400)

        self.f2_1 = Frame(self.f2, width=850, height=400, highlightbackground='black', highlightthickness=2, padx=89,pady=38)
        self.f2_1.pack(pady=15)
        self.f2_1.propagate(False)
        self.f2_1.columnconfigure(index=2, minsize=100)

        self.f2_2 = Frame(self.f2, width=850, highlightbackground='black', highlightthickness=2, padx=73,pady=38)
        self.f2_2.pack()
        self.f2_2.columnconfigure(index=1, minsize=100)
        self.f2_2.columnconfigure(index=2, minsize=150)

        self.canvas = Canvas(self.f1_2, width=self.width, height=self.height)
        self.canvas.pack(padx=100)
        self.image_container = self.canvas.create_image(self.width/2, self.height/2)

    def write_to_db(self,status):
        key, value = list(self.dict_state.keys()), list(self.dict_state.values())
        point = (
            influxdb_client.Point("measurement")
                .tag("camera_id", 0)
                .field(key[0], value[0]*1)
                .field(key[1], value[1]*1)
                .field(key[2], value[2])
                .field(key[3], value[3]*1)
                .field(key[4], value[4])
                .field(key[5], value[5])
                .field(key[6], value[6])
                .field(key[7], value[7])
                .field(key[8], value[8])
                .field(key[9], value[9])
                .field(key[10], value[10])
                .field(key[11], value[11])
                .field(key[12], value[12]*1)
                .field('status', status)
                .time(datetime.utcnow(), influxdb_client.WritePrecision.S)
        )
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)

    def show_status(self, fps_value, status):
        if self.fps_label:
            self.fps_label.destroy()
        if self.status_label:
            self.status_label.destroy()
        self.fps_label = ttk.Label(self.f1_1, text='FPS: '+"{:.2f}".format(fps_value), font=40)
        self.fps_label.pack(anchor='nw')
        self.status_label = ttk.Label(self.f1_1, text='Status: '+status, font=40)
        self.status_label.pack(anchor='nw')

    def get_dict_state(self,dict_state,flag):
        self.dict_state = dict_state
        if flag:
            for i, (key, values) in enumerate(self.dict_state.items()):
                if self.dict_state_qualities[key] == self.TF:
                    self.dict_state_renamed[i] = BooleanVar(0)
                elif self.dict_state_qualities[key] == self.numerical:
                    self.dict_state_renamed[i] = DoubleVar(0)
                elif self.dict_state_qualities[key] == self.word:
                    self.dict_state_renamed[i] = StringVar(0)

    def create_labels(self, flag):
        for widget in self.f2_2.winfo_children():
            widget.destroy()
        temp = None
        if flag:
            for i, (key, value) in enumerate(self.dict_state.items()):
                if self.dict_state_qualities[key] == self.numerical:
                    if key == 'blink time':
                        color = 'green'
                    elif key == 'EAR':
                        color = 'green' if value >= self.dict_state_renamed[i].get() else 'red'
                    else:
                        value = m.fabs(value)
                        color = 'green' if value <= self.dict_state_renamed[i].get() else 'red'
                    temp ="{:05.2f}".format(value)
                elif self.dict_state_qualities[key] == self.TF:
                    temp = ' True' if value else 'False'
                    if key == 'mp_detection':
                        color = 'green'
                    else:
                        color = 'green' if not(value) else 'red'
                else:
                    if key == 'emotion':
                        color = 'green' if (value in ['neutral', 'happy', 'sad']) else 'red'
                    else:
                        color = 'green' if (value == 'forward') else 'red'
                    temp = value
                ttk.Label(self.f2_2, text=key, font=(30)).grid(row=i, column=0, sticky='w')
                ttk.Label(self.f2_2, text=temp, font=(30), foreground=color).grid(row=i,
                                                                                                              column=2,
                                                                                                              sticky='e')
        else:
            color = 'red'
            for i in range(14):
                ttk.Label(self.f2_2, text=list(self.dict_state_qualities.keys())[i], font=(30)).grid(row=i, column=0, sticky='w')
                if list(self.dict_state_qualities.keys())[i] == 'mp_detection':
                    ttk.Label(self.f2_2, text='False', font=(30), foreground=color).grid(row=i,
                                                                                      column=2,
                                                                                      sticky='e')
                else:
                    ttk.Label(self.f2_2, text='No data', font=(30), foreground=color).grid(row=i,
                                                                                         column=2,
                                                                                         sticky='e')

    def change(self, newVal, i, key):
        if (key == 'blink time'):
            self.label_list[i]["text"] = f"{float(newVal):04.2f}"
        elif (key == 'EAR'):
            self.label_list[i]["text"] = f"{float(newVal):04.3f}"
        else:
            self.label_list[i]["text"] = f"{float(newVal):5.0f}"

    def create_slider(self, default_value, left_border, right_border, count, i, key):
        slider = ttk.Scale(self.f2_1, orient=HORIZONTAL, length=200, from_=left_border, to=right_border,
                           variable=self.dict_state_renamed[i],
                           command=lambda newValue, i=count: self.change(newValue, i, key))
        slider.set(default_value)
        slider.grid(row=i, column=1)

    def create_scales(self, flag):
        if flag:
            count = 0
            for i, (key, values) in enumerate(self.dict_state.items()):
                if self.dict_state_qualities[key] == self.numerical:
                    label = ttk.Label(self.f2_1, text="{:3.0f}".format(float(self.dict_state_renamed[i].get())),
                                      font=30)
                    label.grid(row=i, column=2)
                    self.label_list.append(label)
                    ttk.Label(self.f2_1, text=key, font=(30)).grid(row=i, column=0, sticky='w')
                    if key in ['x head', 'y head', 'z head']:
                        self.create_slider(20, 15, 25, count, i, key)
                    elif key =='blink time':
                        self.create_slider(3,2.4,4,count,i, key)
                    elif key == 'num fast blinks':
                        self.create_slider(20,15,30,count,i, key)
                    elif key == 'drowsy time':
                        self.create_slider(10, 5, 20, count, i, key)
                    elif key == 'EAR':
                        self.create_slider(0.225, 0.1, 0.25, count, i, key)
                    else:
                        self.create_slider(30, 20, 40, count, i, key)
                    count+=1

    def running_loop(self, image_info, dict_state,status, flag1, fps_value, flag2):
        self.get_dict_state(dict_state,flag1)
        if not(dict_state['mp_detection']):
            self.dict_state = {'mp_detection': False, 'want to sleep / stressed': True, 'num fast blinks': 0,
                            'is sleeping': False, 'drowsy time': 0, 'blink time': 0,
                            'EAR': 0, 'emotion': 'sad',
                            'position': 'forward', 'x head': 0,
                            'y head': 0, 'z head': 0, "looking away":0, "don't look": True}
            self.get_dict_state(self.dict_state,flag1)
        if flag2:
            self.write_to_db(status)
        img_rgb = cv2.cvtColor(image_info, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.show_status(fps_value, status)
        self.canvas.itemconfig(self.image_container, image=img_tk)
        self.create_labels(dict_state['mp_detection'])
        self.create_scales(flag1)
        self.app.update()
        return self.dict_state_renamed