from openvino.inference_engine import IECore
import time
import cv2 as cv
import wave
import pyaudio
import win32com.client
import pygame
import time
import numpy as np
import os
import serial #导入串口通信库
from playsound import playsound
from pygame import mixer

# 语音合成输出
def speak(s):
    print("-->" + s)
    win32com.client.Dispatch("SAPI.SpVoice").Speak(s)

def ssd_video_demo():
    ie = IECore()
    for device in ie.available_devices:
        print(device)
    ser = serial.Serial(port="COM9", baudrate=9600)

    ser.port = 'COM9'
    ser.baudrate = 9600
    ser.stopbits = 1
    ser.bytesize = 8
    ser.parity = 'N'
    while (ser.isOpen() == 0):
        print('打开串口失败')
        ser.open()

    #cars
#   model_xml = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-detection-0200/FP32/person-detection-0200.xml"
#  model_bin = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-detection-0200/FP32/person-detection-0200.bin"
    model_xml = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-2001/FP32/person-vehicle-bike-detection-2001.xml"
    model_bin = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-2001/FP32/person-vehicle-bike-detection-2001.bin"
#    model_xml = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml"
#    model_bin = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.bin"
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    print(out_blob)


    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)

    cap = cv.VideoCapture('D:/openvinofile/loadd.mp4')
    exec_net = ie.load_network(network=net, device_name="CPU")
    yanshi=0
    while True:
        ret, frame = cap.read()#每帧
        # frame = cv.GaussianBlur(frame, (3, 3), 0, 0)
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))
        image = image.transpose(2, 0, 1)
        #print(image.shape)
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob: [image]})
        inf_end = time.time() - inf_start
        #print("infer time(ms)：%.3f" % (inf_end * 1000))
        #print(out_blob)
        ih, iw, ic = frame.shape
        res = res[out_blob]
        s = 0
        i=0
        j=0
        m=0
        yanshi=yanshi+1
        for obj in res[0][0]:#每个框
            if obj[1]==0:#vehicle
                if obj[2] > 0.5:
                    xmin = int(obj[3] * iw)
                    ymin = int(obj[4] * ih)
                    xmax = int(obj[5] * iw)
                    ymax = int(obj[6] * ih)
                    s = ymax - ymin
                    n = 0.5 * ih/ s
                    if n < 1:
                        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2, 8)#红色
                        # ser.write("1\r\n".encode('utf-8'))
                        # print("1")
                        i=1

                    if 1 < n :
                        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2, 8)#蓝色
                    cv.putText(frame, str(obj[2]), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
            if obj[1] == 1:  # 人
                if obj[2] > 0.5:
                    xmin = int(obj[3] * iw)
                    ymin = int(obj[4] * ih)
                    xmax = int(obj[5] * iw)
                    ymax = int(obj[6] * ih)
                    s = ymax - ymin
                    #print("s=", s )
                    #print("h=", h )
                    n = 1.0 * ih/ s
                    #print("n=",n)
                    if n < 5:
                        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2, 8)
                        # ser.write("2\r\n".encode('utf-8'))
                        j=1
                    if 5 < n :
                        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2, 8)

                    cv.putText(frame, str(obj[2]), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
            if obj[1] == 2:  # 自行车
                if obj[2] > 0.5:
                    xmin = int(obj[3] * iw)
                    ymin = int(obj[4] * ih)
                    xmax = int(obj[5] * iw)
                    ymax = int(obj[6] * ih)
                    s = ymax - ymin
                    #print("s=", s )
                    #print("h=", h )
                    n = 1.0 * ih/ s
                    #print("n=",n)
                    if n < 5:
                        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2, 8)
                        # ser.write("3\r\n".encode('utf-8'))
                        m=1
                    if 5 < n :
                        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2, 8)

                    cv.putText(frame, str(obj[2]), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
        print (j,m,i)
        # if alcohol_num=="8\r\n":
        #             ser.write(b"2")
        #             sleep(1)
        #             print('ok')
        if j == 1 and yanshi>100:
            ser.write(b"3")
            #time.sleep(2)
            yanshi=0
        if j==0 and m==1 and yanshi>100:
            ser.write(b"2")
           # time.sleep(2)
            yanshi = 0
        if j==0 and m==0 and i==1 and yanshi>100:
            ser.write(b"1")
            print('ok')
            #time.sleep(2)
            yanshi = 0

        cv.putText(frame, "infer time(ms): %.3f, FPS: %.2f" % (inf_end * 1000, 1 / (inf_end+float("1e-8"))), (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
        cv.namedWindow("Pedestrian Detection", 0)
        #print(frame.shape)
        cv.imshow("Pedestrian Detection", frame)
        c = cv.waitKey(20)
        if c == 27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()
def music():
    ser = serial.Serial(port="COM9", baudrate=9600)

    ser.port = 'COM9'
    ser.baudrate = 9600
    ser.stopbits = 1
    ser.bytesize = 8
    ser.parity = 'N'
    while (ser.isOpen() == 0):
        print('打开串口失败')
        ser.open()
    while 1:
        num=ser.readline().decode()
        print(num)
        if num=='9\r\n':
            mixer.init()  # 初始化混音器模块
            mixer.music.load(r'C:\Users\Administrator\Desktop\西安人的歌.mp3')  # 载入待播放音乐文件
            mixer.music.play()  # 开始播放音乐流
        if num=='8\r\n':
            mixer.init()  # 初始化混音器模块
            mixer.music.load(r'C:\Users\Administrator\Desktop\消愁.mp3')  # 载入待播放音乐文件
            mixer.music.play()  # 开始播放音乐流
        if num == '7\r\n':
            mixer.music.stop()  # 结束音乐播放

if __name__ == "__main__":
    ssd_video_demo()

