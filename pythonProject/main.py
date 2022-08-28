from untitled2 import *
from UI_final import *
from alcohol import *
from tired import *
from camera import *
from api import *
from people import *
from car import *
from warn import *
import datetime
import tkinter as tk
from openvino.inference_engine import IECore
import time
import cv2 as cv
import numpy as np
import wave
import pykeyboard as keyboard
import pyaudio
from aip import AipSpeech
import dlib
from scipy.spatial import distance
from scipy.spatial import distance as dis
import serial #导入串口通信库
from PIL import Image, ImageDraw, ImageFont
import multiprocessing
from PyQt5.QtWidgets import  QApplication,QMainWindow
from PyQt5 import QtGui, QtCore
import sys
import json
import requests
import wave
from playsound import playsound
# import pygame
import win32com.client
from playsound import playsound
from pygame import mixer
from unittest import result
from aip import AipBodyAnalysis
from numpy import var
from sqlalchemy import true
from threading import Thread
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import math
from threading import Thread


emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
tag2=0

def speak(s):
    print("-->" + s)
    win32com.client.Dispatch("SAPI.SpVoice").Speak(s)
""" 你的 APPID AK SK """
#可修改的数据
""" 你的 APPID AK SK """
# APP_ID = '26752080'
# API_KEY = 'YcYCYLLggUMRyuegdbKGjl5S'
# SECRET_KEY = 'nPUE7eWx4jdixjwMluR6mkqdwxeRQWFE'
# APP_ID = '26833220'
# API_KEY = 'VXACMXeqofzPIfNHsTqSn0f0'
# SECRET_KEY ='p24qognFY0OLaAsvVUCstZG3WjtGQRxX'

# APP_ID = '26837070'
# API_KEY = 'GuyzvF0M7OWnVibftfiYh6qj'
# SECRET_KEY ='cVzOjUMuluOaI7uoMcf3dsmFu5g8Lw7M'

APP_ID = '26837140'
API_KEY = 'LhihybKkEACBdtEXy3ms2EFG'
SECRET_KEY ='OEQMnnnbQIkIUYvKwg8Ey0jH1jm1oKtm'

tired_click=0
outside_click=0
tag=0

# 调用人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/openvinofile/shape_predictor_68_face_landmarks.dat")
#串口定义
alcohol_num='456'
date='123'
playstate=0
# 设定人眼标定点
LeftEye_Start = 36
LeftEye_End = 41
RightEye_Start = 42
RightEye_End = 47
Mouth_Start = 48
Mouth_End = 59
Radio=0.22  #横纵比阈值
#Radio=100
Low_radio_constant = 30  #意味着连续多少帧横纵比小于Radio小于阈值时，判断疲劳
Low_radio_constant_mouth=15  #意味着连续多少帧横纵比小于Radio小于阈值时，判断疲劳
Mouth_Moude_Radio=1.0 #嘴部纵横比阈值
HAR_THRESH=0.3 #头部姿态阈值
NOD_AR_CONSEC_FRAMES=3 #帧数
hCOUNTER=0

# 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                         [1.330353, 7.122144, 6.903745],  # 29左眉右角
                         [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                         [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                         [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                         [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                         [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                         [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                         [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                         [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                         [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                         [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                         [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])  # 6下巴角

# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):  # 头部姿态估计
    # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reprojectdst, _ = cv.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    return reprojectdst, euler_angle  # 投影误差，欧拉角


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

def tiredetection(x):
    print('tireding')
    alarm = False
    mouth_alarm = False  # 初始化嘴巴警报
    frame_counter = 0  # 连续帧计数
    frame_counter_mouth = 0
    mouth = 0
    eye = 0
    yanshi3=0
    yanshi2=0
    head=0
    hCOUNTER=0
    hTOTAL=0
    ie = IECore()
    model_xml = "D:/openvinofile/face-detection-0102.xml"
    model_bin = "D:/openvinofile/face-detection-0102.bin"
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.input_info[input_blob].input_data.shape
    exec_net = ie.load_network(network=net, device_name="CPU")
    # 加载人脸表情识别模型
    em_xml = "D:/openvinofile/emotions-recognition-retail-0003.xml"
    em_bin = "D:/openvinofile/emotions-recognition-retail-0003.bin"
    em_net = ie.read_network(model=em_xml, weights=em_bin)
    em_input_blob = next(iter(em_net.input_info))
    em_out_blob = next(iter(em_net.outputs))
    en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
    em_exec_net = ie.load_network(network=em_net, device_name="CPU")
    print('ok1')
    while True:
        with open("start.txt", "r") as f:
            data = f.readline()
        if data=='1':
            break
    cap=cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()  # 读取每一帧
        with open("start.txt", "r") as f:
            data = f.readline()
        if data=='1':
            ret=True
        elif data=='0':
            ret=False
        frame = cv.flip(frame, 1)

        if ret:
            with open("emotion.txt","r") as f:
                data=f.readline()
            if data=='0':
                print('start')
                yanshi2 = yanshi2 + 1
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



                    # 头部姿态
                    shape = face_utils.shape_to_np(shape)
                    reprojectdst, euler_angle = get_head_pose(shape)
                    har = euler_angle[0, 0]  # 取pitch旋转角度
                    if har > HAR_THRESH:  # 点头阈值0.3
                        hCOUNTER += 1
                    else:
                        # 如果连续3次都小于阈值，则表示瞌睡点头一次
                        if hCOUNTER >= NOD_AR_CONSEC_FRAMES:  # 阈值：3
                            hTOTAL += 1
                        # 重置点头帧计数器
                        hCOUNTER = 0
                    if hTOTAL==3:
                        hTOTAL = 0
                        head = 1
                    else:
                        head = 0
                    if (mouth_alarm == True or alarm == True or head == True) and yanshi2>100:
                        with open("tired_alarm.txt", "w") as f:
                            f.write("1")
                        mixer.init()  # 初始化混音器模块
                        mixer.music.load(r'C:\Users\Administrator\Desktop\疲劳驾驶.mp3')  # 载入待播放音乐文件
                        mixer.music.play()  # 开始播放音乐流
                        yanshi2=0
                    else:
                        with open("tired_alarm.txt", "w") as f:
                            f.write("0")

                    # for start, end in line_pairs:
                    #     cv.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                    #
                    for (x, y) in shape:
                        cv.circle(frame, (x, y), 1, (0, 0, 255), -1)

                    # 显示结果
                    if mouth == 1 and eye == 1:
                        cv.putText(frame, "serious", (40, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif mouth == 1 or eye == 1 or head==1:
                        cv.putText(frame, "middle", (40, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv.putText(frame, "normal", (40, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv.putText(frame, "Eye Ratio{:.2f}".format(mean_Ratio), (200, 70),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)
                    cv.putText(frame, "Mouth Ratio{:.2f}".format(Mouth_Ratio), (200, 40),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)
                    cv.putText(frame, "HEAD number{:.2f}".format(hTOTAL), (200, 100),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)
                # 界面显示
                with open("tired_change.txt", "r") as f:
                    data = f.readline()
                if data=='1':
                    cv.imshow('tired',frame)
                else :
                    cv.destroyAllWindows()
            elif data=='1':
                yanshi3 = yanshi3 + 1
                image = cv.resize(frame, (w, h))
                image = image.transpose(2, 0, 1)
                inf_start = time.time()
                res = exec_net.infer(inputs={input_blob: [image]})
                inf_end = time.time() - inf_start
                # print("infer time(ms)：%.3f"%(inf_end*1000))
                ih, iw, ic = frame.shape
                res = res[out_blob]
                for obj in res[0][0]:
                    if obj[2] > 0.75:
                        xmin = int(obj[3] * iw)
                        ymin = int(obj[4] * ih)
                        xmax = int(obj[5] * iw)
                        ymax = int(obj[6] * ih)
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if xmax >= iw:
                            xmax = iw - 1
                        if ymax >= ih:
                            ymax = ih - 1
                        roi = frame[ymin:ymax, xmin:xmax, :]
                        roi_img = cv.resize(roi, (ew, eh))
                        roi_img = roi_img.transpose(2, 0, 1)
                        em_res = em_exec_net.infer(inputs={em_input_blob: [roi_img]})
                        prob_emotion = em_res[em_out_blob].reshape(1, 5)
                        label_index = np.argmax(prob_emotion, 1)
                        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                        cv.putText(frame, "infer time(ms): %.3f" % (inf_end * 1000), (50, 50),
                                   cv.FONT_HERSHEY_SIMPLEX, 1.0,
                                   (255, 0, 255),
                                   2, 8)
                        cv.putText(frame, emotions[np.int(label_index)], (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX,
                                   0.55,
                                   (0, 0, 255),
                                   2, 8)
                        if emotions[np.int(label_index)]=="anger" and yanshi3>100:
                            # if tag2==0:
                            #     tag2=1
                            mixer.init()  # 初始化混音器模块
                            mixer.music.load(r'C:\Users\Administrator\Desktop\路怒.mp3')  # 载入待播放音乐文件
                            mixer.music.play()  # 开始播放音乐流
                            yanshi3=0
                        # elif emotions[np.int(label_index)]=="neutral" or emotions[np.int(label_index)]=="happy":
                        #     tag2=0
                cv.imshow("Face+emotion Detection", frame)
                c = cv.waitKey(1)
                if c == 27:

                    break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def outsidetection(x):
    ie = IECore()
    for device in ie.available_devices:
        print(device)
    print("ok3")
    model_xml = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-2001/FP32/person-vehicle-bike-detection-2001.xml"
    model_bin = "C:/Program Files (x86)/Intel/openvino_2021.4.752/deployment_tools/open_model_zoo/tools/downloader/intel/person-vehicle-bike-detection-2001/FP32/person-vehicle-bike-detection-2001.bin"
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.input_info[input_blob].input_data.shape
    exec_net = ie.load_network(network=net, device_name="CPU")
    yanshi=0
    print('ok2')
    while (True):
        with open("outside.txt", "r") as f:
            data = f.readline()
        if data == '1':
            break
    cap = cv.VideoCapture("D:/openvinofile/loadd.mp4")
    while True:
        ret, frame = cap.read()#每帧
        with open("start.txt", "r") as f:
            data = f.readline()
        if data == '0':
            ret=False
        print('outside')
        while ret is False:
            print(1)
            ret, frame = cap.read()  # 每帧
            with open("start.txt", "r") as f:
                data = f.readline()
            if data == '0':
                ret = False
        image = cv.resize(frame, (w, h))
        image = image.transpose(2, 0, 1)
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob: [image]})
        inf_end = time.time() - inf_start
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
        if j == 1 and yanshi>200:
            mixer.init()  # 初始化混音器模块
            mixer.music.load(r'C:\Users\Administrator\Desktop\行人.mp3')  # 载入待播放音乐文件
            mixer.music.play()  # 开始播放音乐流
            # with open("people_alarm.txt","w") as f:
            #     f.write("1")
            print("peoplepeoplepeoplepeoplepeoplepeoplepeoplepeople")
            yanshi=0
        elif j==0 and m==1 and yanshi>200:
            mixer.init()  # 初始化混音器模块
            mixer.music.load(r'C:\Users\Administrator\Desktop\车辆.mp3')  # 载入待播放音乐文件
            mixer.music.play()  # 开始播放音乐流
            # ser.write(b"2")
            # with open("car_alarm.txt","w") as f:
            #     f.write("1")
            yanshi = 0
        elif j==0 and m==0 and i==1 and yanshi>200:
            mixer.init()  # 初始化混音器模块
            mixer.music.load(r'C:\Users\Administrator\Desktop\车辆.mp3')  # 载入待播放音乐文件
            mixer.music.play()  # 开始播放音乐流
            # ser.write(b"2")
            # with open("car_alarm.txt","w") as f:
            #     f.write("1")
            print("carcarcarcarcarcarcarcarcarcarcarcarcarcarcarcar")
            yanshi = 0
        elif j==0 and m==0 and i==0 :
            with open("people_alarm.txt", "w") as f:
                f.write("0")
            with open("car_alarm.txt","w") as f:
                f.write("0")

        cv.putText(frame, "infer time(ms): %.3f, FPS: %.2f" % (inf_end * 1000, 1 / (inf_end+float("1e-8"))), (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
        with open("outside_change.txt", "r") as f:
            data = f.readline()
        if data == '1':
            cv.imshow("Pedestrian Detection", frame)
        else:
            cv.destroyAllWindows()
        c = cv.waitKey(10)
        if c == 27:
            break

def actiondetection(x):
    """ 需要的参数（0则为需要，1则为不需要） """
    smoke = 0  # 吸烟
    cellphone = 0  # 打手机
    not_buckling_up = 1  # 未系安全带
    both_hands_leaving_wheel = 1  # 双手离开方向盘
    not_facing_front = 1  # 视角未看前方
    """ 可按实际情况修改 """
    """ "1"，表示左舵车 "0"，表示右舵车 """
    wheel_location = 1
    client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)  # 生成key
    number = 1
    timee=0
    while True:
        with open("start.txt", "r") as f:
            data = f.readline()
        if data == '1':
            break
    video = cv2.VideoCapture(1)
    while True:
        ret, frame = video.read()
        with open("start.txt", "r") as f:
            data = f.readline()
        if data=='1':
            ret=True
        elif data=='0':
            ret=False
        # cv.imshow("frame",frame)
        # image=get_file_content(frame)
        while ret is False:
            ret, frame = video.read()
            with open("start.txt", "r") as f:
                data = f.readline()
            if data == '1':
                ret = True
            elif data == '0':
                ret = False
        image = cv2.imencode('.jpg', frame)[1]
        """ 调用驾驶行为分析 """
        client.driverBehavior(image)
        """ 如果有可选参数 """
        options = {}
        timee=timee+1
        options_str = ""
        if smoke == 0: options_str = options_str + "smoke,"
        if cellphone == 0: options_str = options_str + "cellphone,"
        if not_buckling_up == 0: options_str = options_str + "not_buckling_up,"
        if both_hands_leaving_wheel == 0: options_str = options_str + "both_hands_leaving_wheel,"
        if not_facing_front == 0: options_str = options_str + "not_facing_front,"
        options["type"] = options_str
        options["wheel_location"] = wheel_location
        print("*************")

        """ 带参数调用驾驶行为分析 """
        a = client.driverBehavior(image, options)
        # print(json.dumps(a,indent=3)) #格式化输出
        data = json.dumps(a)
        data_dict = json.loads(data)
        # result=json.loads()
        data2 = data_dict["person_info"]
        if data2!=[]:
            data3 = data2[0]
            data4 = data3['attributes']
            data5 = data4['smoke']
            smoke_score = data5['score']
            data6 = data4['cellphone']
            phone_score = data6['score']
            if (smoke_score > 0.48 and timee>1):
                print("正在抽烟")
                with open("action_alarm.txt","w") as f:
                    f.write("1")
                mixer.init()  # 初始化混音器模块
                mixer.music.load(r'C:\Users\Administrator\Desktop\吸烟.mp3')  # 载入待播放音乐文件
                mixer.music.play()  # 开始播放音乐流
                timee = 0
            elif (phone_score > 0.69 and timee>1):
                print("正在打电话")
                with open("action_alarm.txt","w") as f:
                    f.write("1")
                mixer.init()  # 初始化混音器模块
                mixer.music.load(r'C:\Users\Administrator\Desktop\电话.mp3')  # 载入待播放音乐文件
                mixer.music.play()  # 开始播放音乐流
                timee = 0
            else :
                with open("action_alarm.txt","w") as f:
                    f.write("0")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()



# def alcohol_detect():
#     global alcohol_num
#     ser.port='COM9'
#     ser.baudrate=9600
#     ser.stopbits=1
#     ser.bytesize=8
#     ser.parity='N'
#     while (ser.isOpen()==0):
#         print('打开串口失败')
#         ser.open()
#     alcohol_num=ser.readline().decode('utf-8')
#     print('ok')


def actdetection():
    print('actdetection')
    warnaction()
    

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.ser = serial.Serial(port="COM9", baudrate=9600, stopbits=1, bytesize=8, parity='N')
        self.ui=Login_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton.clicked.connect(self.go_to_inter)
        # timer = QtCore.QTimer()
        # self.timer = timer
        # self.timer.start(500)
        # self.timer.timeout.connect(self.timer)
        self.show()


    def timer(self):
        print(1)
        # self.num = self.ser.readline(self.ser.inWaiting())
        # print(self.num)
        # if self.num.decode('utf-8')=='3\r\n':
        #     with open("ready.txt","w") as f:
        #         f.write("1")

    def go_to_inter(self):
        account=self.ui.lineEdit.text()
        password=self.ui.lineEdit_2.text()
        if account=='1' and password=='2':
            print(main_camera())
            with open("start.txt", "w") as f:
                f.write("1")
            InterWindow()
            self.close()
        else :
            pass



    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

class InterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.r = requests.get('http://www.weather.com.cn/data/sk/101110101.html')
        self.r.encoding = 'utf-8'
        self.ser = serial.Serial(port="COM9", baudrate=9600,stopbits=1,bytesize=8,parity='N',timeout=0.5)
        while (self.ser.isOpen() == 0):
            print('打开串口失败')
            self.ser.open()
        self.alcohol=False
        self.count=0
        self.flag=0
        self.tag=0
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton_tired.clicked.connect(change_tired)
        self.ui.pushButton_road.clicked.connect(change_outside)
        self.ui.pushButton_action.clicked.connect(actdetection)
        self.ui.pushButton_exit.clicked.connect(exitanger)
        self.ui.pushButton_music.clicked.connect(playmusic)
        self.ui.pushButton_chick.clicked.connect(changemusic)
        self.ui.pushButton_emotion.clicked.connect(emotionstart)
        self.ui.pushButton_alcohol.setText('酒精检测\n  正常')
        self.ui.pushButton_weather.clicked.connect(lambda: self.ui.label.setText('请勿酒驾'))
        self.i = datetime.datetime.now()
        self.ui.pushButton_weather.setText('西安\n'+str(self.i.month)+'月'+str(self.i.day)+'日\n'+self.r.json()['weatherinfo']['temp']+'°C '+self.r.json()['weatherinfo']['WD'])
        timer=QtCore.QTimer()
        self.timer=timer
        self.timer.start(300)
        self.timer.timeout.connect(self.clicked_button)
        self.show()

    def clicked_button(self):
        self.i=datetime.datetime.now()
        print(type(self.i.hour))
        date=str(self.i.hour)
        print(type(date))
        # alcohol_num=self.ser.readline(self.ser.inWaiting())
        self.ui.label.setText(str(self.i.hour)+':'+str(self.i.minute))

        # if self.alcohol==False and alcohol_num[9:12]>='500' :
        #     self.alcohol=True
        #     print('123')
        #     TiredWindow()
        self.music()
        with open("people_alarm.txt","r") as f:
            if f.readline()=='1':
                print('recievereviiiiiiiiiiiiiiiiiiiiii')
                mixer.init()  # 初始化混音器模块
                mixer.music.load(r'C:\Users\Administrator\Desktop\行人.mp3')  # 载入待播放音乐文件
                mixer.music.play()  # 开始播放音乐流
                with open("people_alarm.txt", "w") as f:
                    f.write("0")
                PeopleWindow()
        with open("car_alarm.txt","r") as f:
            if f.readline()=='1':
                print('recievereviiiiiiiiiiiiiiiiiiiiii')
                # self.ser.write('2')
                # CarWindow()
                mixer.init()  # 初始化混音器模块
                mixer.music.load(r'C:\Users\Administrator\Desktop\车辆.mp3')  # 载入待播放音乐文件
                mixer.music.play()  # 开始播放音乐流
                with open("car_alarm.txt", "w") as f:
                    f.write("0")
                CarWindow()
        with open("tired_alarm.txt","r") as f:
            if f.readline()=='1':
                TiredWindow()
        with open("action_alarm.txt","r") as f:
            if f.readline()=='1':
                WarnWindow()

    def music(self):
        # self.ser.flushInput()
        self.num = self.ser.readline(self.ser.inWaiting())
        print(self.num)
        self.count+=1
        print(self.count)
        if self.count>=2 :
            if self.num.decode('utf-8') == '9\r\n':
                if self.flag==0:
                    mixer.init()  # 初始化混音器模块
                    mixer.music.load(r'C:\Users\Administrator\Desktop\1.mp3')  # 载入待播放音乐文件
                    mixer.music.play()  # 开始播放音乐流
                    self.flag=1
            elif self.num.decode('utf-8') == '8\r\n':
                if self.flag==1:
                    mixer.init()  # 初始化混音器模块
                    mixer.music.load(r'C:\Users\Administrator\Desktop\2.mp3')  # 载入待播放音乐文件
                    mixer.music.play()  # 开始播放音乐流
                    self.flag = 0
            elif self.num.decode('utf-8') == '7\r\n':
                self.flag=0
                mixer.music.stop()  # 结束音乐播放
            elif self.num.decode('utf-8') == 'n\r\n':
                self.flag=0
                self.ui.pushButton_alcohol.setText('酒精检测\n正常')
                self.tag=0
            elif self.num.decode('utf-8') == 'w\r\n':
                self.flag=0
                self.ui.pushButton_alcohol.setText('酒精检测\n浓度过高')
                if self.tag==0:
                    mixer.init()  # 初始化混音器模块
                    mixer.music.load(r'C:\Users\Administrator\Desktop\就.mp3')  # 载入待播放音乐文件
                    mixer.music.play()  # 开始播放音乐流
                    self.tag=1

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

class AlcoholWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Alcohol_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton.clicked.connect(lambda :self.ui.label.setText('请勿酒驾'))
        self.show()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

class TiredWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Tired_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton.clicked.connect(lambda :self.ui.label.setText('请勿疲劳驾驶'))
        timer = QtCore.QTimer()
        self.timer=timer
        self.timer.start(500)
        self.timer.timeout.connect(self.tired_time)
        self.show()

    def tired_time(self):
        with open("tired_alarm.txt","r") as f:
            if f.readline()=='0':
                self.close()



class CarWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Tired_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.label.setText('注意车辆')
        self.ui.pushButton.clicked.connect(lambda :self.ui.label.setText('注意车辆'))
        timer = QtCore.QTimer()
        self.timer=timer
        self.timer.start(500)
        self.timer.timeout.connect(self.car_time)
        self.show()

    def tired_time(self):
        with open("car_alarm.txt","w") as f:
            f.write("0")
        self.close()


class PeopleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Tired_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.label.setText('注意行人')
        self.ui.pushButton.clicked.connect(lambda :self.ui.label.setText('注意行人'))
        timer = QtCore.QTimer()
        self.timer=timer
        self.timer.start(500)
        self.timer.timeout.connect(self.people_time)
        self.show()

    def people_time(self):
        with open("people_alarm.txt", "w") as f:
            f.write("0")
        self.close()

class WarnWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Tired_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.label.setText('请勿危险驾驶')
        self.ui.pushButton.clicked.connect(lambda: self.ui.label.setText('请勿危险驾驶'))
        timer = QtCore.QTimer()
        self.timer=timer
        self.timer.start(500)
        self.timer.timeout.connect(self.tired_time)
        self.show()

    def tired_time(self):
        with open("tired_alarm.txt","r") as f:
            if f.readline()=='0':
                self.close()

def GUI(x):
    app = QApplication(sys.argv)
    win=LoginWindow()
    app.exec_()


def change_tired():
    global tired_click
    tired_click=1-tired_click
    if tired_click==1:
        with open("tired_change.txt", "w") as f:
            f.write("1")
    else :
        with open("tired_change.txt", "w") as f:
            f.write("0")

def change_outside():
    global outside_click
    outside_click=1+outside_click
    with open("outside.txt","w") as f:
        f.write("1")
    if outside_click%2==1:
        with open("outside_change.txt", "w") as f:
            f.write("1")
        with open("start.txt", "w") as f:
            f.write("1")
    elif outside_click%2==0 and outside_click>1:
        with open("outside_change.txt", "w") as f:
            f.write("0")
        with open("start.txt", "w") as f:
            f.write("0")

def playmusic():
    global playstate
    if playstate==0:
        print("playplay!!!!!!!!!!!!!!!!!!!!!!!!")
        playstate=1
        mixer.init()  # 初始化混音器模块
        mixer.music.load(r'C:\Users\Administrator\Desktop\1.mp3')  # 载入待播放音乐文件
        mixer.music.play()  # 开始播放音乐流
    elif playstate==1 or playstate==2:
        playstate=0
        mixer.music.stop()  # 结束音乐播放

def changemusic():
    global playstate
    if playstate==2:
        playstate=1
        mixer.init()  # 初始化混音器模块
        mixer.music.load(r'C:\Users\Administrator\Desktop\1.mp3')  # 载入待播放音乐文件
        mixer.music.play()  # 开始播放音乐流
    elif playstate==1:
        playstate=2
        mixer.init()  # 初始化混音器模块
        mixer.music.load(r'C:\Users\Administrator\Desktop\2.mp3')  # 载入待播放音乐文件
        mixer.music.play()  # 开始播放音乐流

def exitanger():
    with open("emotion.txt", "w") as f:
         f.write("0")

def emotionstart():
    with open("emotion.txt","w") as f:
        f.write("1")



if __name__=='__main__':
    print("ok6")
    # alcohol_detect()
    with open("start.txt", "w") as f:
        f.write("0")
    with open("outside.txt","w") as f:
        f.write("0")
    with open("people_alarm.txt", "w") as f:
        f.write("0")
    with open("car_alarm.txt", "w") as f:
        f.write("0")
    with open("action_alarm.txt", "w") as f:
        f.write("0")
    with open("outside_change.txt", "w") as f:
        f.write("0")
    with open("tired_change.txt.txt", "w") as f:
        f.write("0")
    with open("emotion.txt", "w") as f:
        f.write("0")
    with open("ready.txt","w") as f:
        f.write("0")

    # actiondetection()

    pool = multiprocessing.Pool(processes=4)

    pool.apply_async(outsidetection, (1,))
    pool.apply_async(GUI, (2,))
    pool.apply_async(tiredetection, (3,))
    pool.apply_async(actiondetection,(4,))

    pool.close()
    pool.join()


