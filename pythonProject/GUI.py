import tkinter as tk
from openvino.inference_engine import IECore
import time
import cv2 as cv
import numpy as np
import wave
import pykeyboard as keyboard
import pyaudio
from aip import AipSpeech
import win32com.client
import pygame
import dlib
from scipy.spatial import distance
from scipy.spatial import distance as dis
import serial #导入串口通信库
from PIL import Image, ImageDraw, ImageFont
import multiprocessing

#串口定义
ser=serial.Serial()
alcohel_num='123'
root=tk.Tk()

# 调用人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/openvinofile/shape_predictor_68_face_landmarks.dat")

# 设定人眼标定点
LeftEye_Start = 36
LeftEye_End = 41
RightEye_Start = 42
RightEye_End = 47
Mouth_Start = 48
Mouth_End = 59
Radio=0.23  #横纵比阈值
#Radio=100
Low_radio_constant = 30  #意味着连续多少帧横纵比小于Radio小于阈值时，判断疲劳
Low_radio_constant_mouth=15  #意味着连续多少帧横纵比小于Radio小于阈值时，判断疲劳
Mouth_Moude_Radio=1.0 #嘴部纵横比阈值


def calculate_Ratio(eye):
    """
    计算眼睛横纵比
    """
    d1 = distance.euclidean(eye[1], eye[5])
    d2 = distance.euclidean(eye[2], eye[4])
    d3 = distance.euclidean(eye[0], eye[3])
    ratio = (d1 + d2) / (2 * d3)
    return ratio

def mouthRatio(mouth):
    """
    计算嘴巴横纵比
    """
    left=dis.euclidean(mouth[2],mouth[10])
    mid=dis.euclidean(mouth[3],mouth[9])
    right=dis.euclidean(mouth[4],mouth[8])
    horizontal=dis.euclidean(mouth[0],mouth[6])
    return 10.0*horizontal/(3.0*left+4.0*mid+3.0*right)

def outsidetection(x):
    print('ssd_demo_detect')
    ie = IECore()
    # cars
    #   model_xml = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-detection-0200/FP32/person-detection-0200.xml"
    #  model_bin = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-detection-0200/FP32/person-detection-0200.bin"
    model_xml = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-2001/FP32/person-vehicle-bike-detection-2001.xml"
    model_bin = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-2001/FP32/person-vehicle-bike-detection-2001.bin"
    #    model_xml = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml"
    #    model_bin = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.bin"
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape

    cap = cv.VideoCapture(1)
    exec_net = ie.load_network(network=net, device_name="CPU")
    while True:
        ret2, frame2 = cap.read()  # 每帧
        # frame = cv.GaussianBlur(frame, (3, 3), 0, 0)
        image = cv.resize(frame2, (w, h))
        image = image.transpose(2, 0, 1)
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob: [image]})
        inf_end = time.time() - inf_start
        print("infer time(ms)：%.3f" % (inf_end * 1000))
        print(out_blob)
        ih, iw, ic = frame2.shape
        res = res[out_blob]
        s = 0
        for obj in res[0][0]:  # 每个框
            if obj[1] == 0:  # vehicle
                if obj[2] > 0.5:
                    xmin = int(obj[3] * iw)
                    ymin = int(obj[4] * ih)
                    xmax = int(obj[5] * iw)
                    ymax = int(obj[6] * ih)
                    s = ymax - ymin
                    n = 0.5 * ih / s
                    if n < 10:
                        cv.rectangle(frame2, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2, 8)  # 红色
                    if 10 < n:
                        cv.rectangle(frame2, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2, 8)  # 蓝色
                    cv.putText(frame2, str(obj[2]), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
            if obj[1] == 1:  # 人
                if obj[2] > 0.5:
                    xmin = int(obj[3] * iw)
                    ymin = int(obj[4] * ih)
                    xmax = int(obj[5] * iw)
                    ymax = int(obj[6] * ih)
                    s = ymax - ymin
                    print("s=", s)
                    print("h=", h)
                    n = 1.0 * ih / s
                    print("n=", n)
                    if n < 10:
                        cv.rectangle(frame2, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2, 8)
                    if 10 < n:
                        cv.rectangle(frame2, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2, 8)

                    cv.putText(frame2, str(obj[2]), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
            if obj[1] == 2:  # 自行车
                if obj[2] > 0.5:
                    xmin = int(obj[3] * iw)
                    ymin = int(obj[4] * ih)
                    xmax = int(obj[5] * iw)
                    ymax = int(obj[6] * ih)
                    s = ymax - ymin
                    print("s=", s)
                    print("h=", h)
                    n = 1.0 * ih / s
                    print("n=", n)
                    if n < 10:
                        cv.rectangle(frame2, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2, 8)
                    if 10 < n:
                        cv.rectangle(frame2, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2, 8)

                    cv.putText(frame2, str(obj[2]), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
        cv.putText(frame2, "infer time(ms): %.3f, FPS: %.2f" % (inf_end * 1000, 1 / (inf_end + float("1e-8"))), (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
        cv.namedWindow("Pedestrian Detection", 0)
        print(frame2.shape)
        cv.imshow("Pedestrian Detection", frame2)
        if cv.waitKey(1)==27:
            break


def tiredetection():
        print('tireding')
        alarm=False
        mouth_alarm = False  # 初始化嘴巴警报
        frame_counter = 0  # 连续帧计数
        frame_counter_mouth = 0
        mouth = 0
        eye = 0
        cap=cv.VideoCapture(0)

        while True:
            print(123)
            ret, frame = cap.read()  # 读取每一帧
            frame = cv.flip(frame, 1)
            if ret:
                print('start')
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                rects = detector(gray, 0)  # 人脸检测

                for rect in rects:
                    shape = predictor(gray, rect)
                    points = np.zeros((68, 2), dtype=int)
                    for i in range(68):
                        points[i] = (shape.part(i).x, shape.part(i).y)

                    # 获取眼睛特征点
                    Lefteye = points[LeftEye_Start: LeftEye_End + 1]
                    Righteye = points[RightEye_Start: RightEye_End + 1]
                    Mouth = points[Mouth_Start: Mouth_End + 1]

                    # 计算眼睛横纵比
                    Lefteye_Ratio = calculate_Ratio(Lefteye)
                    Righteye_Ratio = calculate_Ratio(Righteye)
                    mean_Ratio = (Lefteye_Ratio + Righteye_Ratio) / 2  # 计算两眼平均比例

                    # 计算嘴部横纵比
                    Mouth_Ratio = mouthRatio(Mouth)
                    # print(Mouth_Ratio)

                    # 计算凸包
                    left_eye_hull = cv.convexHull(Lefteye)
                    right_eye_hull = cv.convexHull(Righteye)
                    mouth_hull = cv.convexHull(Mouth)
                    # 绘制轮廓
                    cv.drawContours(frame, [left_eye_hull], -1, [0, 255, 0], 1)
                    cv.drawContours(frame, [right_eye_hull], -1, [0, 255, 0], 1)
                    cv.drawContours(frame, [mouth_hull], -1, [0, 255, 0], 1)

                    # 眨眼判断
                    if mean_Ratio < Radio:
                        frame_counter += 1
                        # print("困了")
                        # print(frame_counter)
                        if frame_counter >= Low_radio_constant:
                            # 发出警报
                            # print("闭眼")
                            eye = 1
                            if not alarm:
                                alarm = True
                            cv.putText(frame, "eye closing", (40, 70),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # print(123456)
                        eye = 0
                        alarm = False
                        frame_counter = 0

                    # 打哈欠判断
                    if Mouth_Ratio < Mouth_Moude_Radio:
                        frame_counter_mouth += 1
                        if frame_counter_mouth >= Low_radio_constant_mouth:
                            # 发出警报
                            mouth = 1
                            if not mouth_alarm:
                                mouth_alarm = True
                            cv.putText(frame, "yawning", (40, 100),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        mouth = 0
                        mouth_alarm = False
                        frame_counter_mouth = 0

                    # 显示结果
                    if mouth == 1 and eye == 1:
                        cv.putText(frame, "serious", (40, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif mouth == 1 or eye == 1:
                        cv.putText(frame, "middle", (40, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv.putText(frame, "normal", (40, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv.putText(frame, "Eye Ratio{:.2f}".format(mean_Ratio), (200, 70),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)
                    cv.putText(frame, "Mouth Ratio{:.2f}".format(Mouth_Ratio), (200, 40),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)
                # 界面显示
                cv.imshow("test", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

def safedetect():
    print('safedetect')

def grade():
    print('grade')

# 语音合成输出
def speaking():
    while(True):
        speak('Haibara AI 为你服务')
        print("请讲话...")

        audio_path = "D:/openvinofile/test1.wav"
        # 录制语音指令
        audio_record(audio_path, 3)

        print("开始做语音识别...")
        ret = audio_discern(audio_path)  # 识别语音指令
        if ret["err_no"] == 0:
            text = ret["result"][0]
            print(text)

            if '道路' in text:
                speak('Haibara AI 为你开启道路检测')
                outsidetection()

            elif '疲劳' in text:
                speak('Haibara AI 为你开启疲劳检测')
                tiredetection()

            elif '安全' in text:
                speak('Haibara AI 为你开启安全检测')
                safedetect()

            elif '车载' in text:
                speak('进入车载导航')
                grade()

            # 如果是"退出"指令则结束程序
            elif text.find("再见") != -1:
                speak('期待下次见面')
                break
            else:
                pass


# 语音合成输出
def speak(s):
    print("-->" + s)
    win32com.client.Dispatch("SAPI.SpVoice").Speak(s)


# 调用百度云，进行语音识别
def audio_discern(audio_path="./test.wav", audio_type="wav"):
    """ 百度云的ID，免费注册 """
    APP_ID = '26307395'
    API_KEY = 'ob6MUzx4pi1qVnL9QZRGa7dM'
    SECRET_KEY = 'MMQ7P9DKbCXEgIpReobk4zTn5vD0q42q'

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    # 读取文件
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    # 识别本地文件
    text = client.asr(get_file_content(audio_path), audio_type, 16000)
    return text


def alcohol_detect():
    global alcohel_num
    global root
    ww = root.winfo_screenwidth()
    wh = root.winfo_screenheight()
    sw = 1000
    sh = 600
    x = (ww - sw) / 2
    y = (wh - sh) / 2
    imgw = 250
    imgh = 150
    insw = (sw - 2 * imgw) / 4
    insh = (sh - 2 * imgh) / 4
    image1 = tk.PhotoImage(file='D:\GUIfile\image1.png')
    btn1 = tk.Button(root, command=alcohol_detect, font=("黑体", 18, "bold"),text=alcohel_num, image=image1, fg='blue',
                     width=imgw,
                     height=imgh, bd=1, compound='center')
    btn1.place(x=insw, y=insh)
    ser.port='COM7'
    ser.baudrate=9600
    ser.stopbits=1
    ser.bytesize=8
    ser.parity='N'
    while (ser.isOpen()==0):
        print('打开串口失败')
        ser.open()
    print('打开串口成功')
    alcohel_num=ser.readline()
    print(alcohel_num)
    btn1.after(500, alcohol_detect)


# 用Pyaudio库录制音频
def audio_record(out_file, rec_time):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # 16bit编码格式
    CHANNELS = 1  # 单声道
    RATE = 16000  # 16000采样频率
    p = pyaudio.PyAudio()
    # 创建音频流
    stream = p.open(format=FORMAT,  # 音频流wav格式
                    channels=CHANNELS,  # 单声道
                    rate=RATE,  # 采样率16000
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Start Recording...")
    frames = []  # 录制的音频流
    # 录制音频数据
    for i in range(0, int(RATE / CHUNK * rec_time)):
        data = stream.read(CHUNK)
        frames.append(data)
    # 录制完成
    # print(frames)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 保存音频文件
    with wave.open(out_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def frpc():
    while (True):
        print(1223)

def frco():
    while(True):
        print(567)



def GUI(x):
    global alcohel_num
    global root
    ww = root.winfo_screenwidth()
    wh = root.winfo_screenheight()
    sw = 1000
    sh = 600
    x = (ww - sw) / 2
    y = (wh - sh) / 2
    imgw = 250
    imgh = 150
    insw = (sw - 2 * imgw) / 4
    insh = (sh - 2 * imgh) / 4
    limg = 75
    root.geometry('%dx%d+%d+%d' % (sw, sh, x, y))
    root.title('安车行控制界面')

    frame1 = tk.Frame(root)
    frame1.pack()

    frame2 = tk.Frame(root)
    frame2.pack()

    image1 = tk.PhotoImage(file='D:\GUIfile\image1.png')
    image2 = tk.PhotoImage(file='D:\GUIfile\image2.png')
    image3 = tk.PhotoImage(file='D:\GUIfile\warning.png')
    image4 = tk.PhotoImage(file='D:\GUIfile\grade.png')
    image5 = tk.PhotoImage(file='D:\GUIfile\speaking.png')
    # tk.Label(root, image=image1, width=800, height=600).place(x=0, y=0)

    imagegif = tk.PhotoImage(file='D:\GUIfile\ground.png')
    menu = tk.Label(root, image=imagegif, width=sw, height=sh)
    menu.place(x=0, y=0)

    btn1 = tk.Button(root, command=alcohol_detect, font=("黑体", 18, "bold"),text=alcohel_num, image=image1, fg='yellow',
                     width=imgw,
                     height=imgh, bd=1, compound='center')
    btn1.place(x=insw, y=insh)

    btn2 = tk.Button(root, command=tiredetection, text='疲劳检测', font=("黑体", 18), image=image2, fg='yellow',
                     width=imgw,
                     height=imgh, bd=1, compound='center')
    btn2.place(x=3 * insw + imgw, y=insh)

    btn3 = tk.Button(root, command=safedetect, text='安全检测', font=("黑体", 18), image=image3, fg='yellow', width=imgw,
                     height=imgh, bd=1, compound='center')
    btn3.place(x=insw, y=3 * insh + imgh)

    btn4 = tk.Button(root, command=grade, text='车载导航', font=("黑体", 18), image=image4, fg='yellow', width=imgw,
                     height=imgh, bd=1, compound='center')
    btn4.place(x=3 * insw + imgw, y=3 * insh + imgh)

    btn5 = tk.Button(root, command=speaking, text='语音助手', anchor='n', bd=0, font=("黑体", 8), image=image5, fg='blue',
                     width=limg,height=limg)
    btn5.place(x=(sw - limg) / 2, y=(sh - limg) / 2)

    root.mainloop()



def plt(x):
    while (True):
        print(123)


if __name__ == '__main__':

    pool = multiprocessing.Pool(processes=3)

    pool.apply_async(GUI, (1, ))
    pool.apply_async(tiredetection, (2, ))
    pool.apply_async(outsidetection,(3, ))

    pool.close()
    pool.join()

