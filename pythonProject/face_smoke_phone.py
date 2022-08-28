import numpy as np
import cv2 as cv
import dlib
from scipy.spatial import distance
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance as dis
from unittest import result
from aip import AipBodyAnalysis
import json
from numpy import var
from sqlalchemy import true
from threading import Thread

# 调用人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/openvinofile/shape_predictor_68_face_landmarks.dat")

APP_ID = '26645471'
API_KEY = 'O5Ua1M3BfUmWjwVhkfn8Lf5b'
SECRET_KEY = 'mlsbqfwaub0DBclt0tsXETPPA4VhyqQ6'

capture = cv.VideoCapture(2)#0为默认摄像头
def camera():
    while True:
        # 获得图片
        ret, frame = capture.read()
        #cv2.imshow("窗口名称", "窗口显示的图像)
        # 显示图片
        #cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

Thread(target=camera).start()  # 引入线程防止在识别的时候卡死

smoke=0 #吸烟
cellphone=0 #打手机
not_buckling_up=1 #未系安全带
both_hands_leaving_wheel=1 #双手离开方向盘
not_facing_front=1 #视角未看前方
""" 可按实际情况修改 """
""" "1"，表示左舵车 "0"，表示右舵车 """
wheel_location=1
client=AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY) #生成key


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
Mouth_Moude_Radio=1.3 #嘴部纵横比阈值


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

def main():
    """
    主函数
    """
    alarm = False  # 初始化眼睛警报
    mouth_alarm=False #初始化嘴巴警报
    frame_counter=0  # 连续帧计数
    frame_counter_mouth=0
    mouth=0
    eye=0
    cap = cv.VideoCapture(0)  # 0摄像头摄像
    number=1
    timeF=50
    while cap.isOpened():
        ret, frame = cap.read()  # 读取每一帧
        frame = cv.flip(frame, 1)
        number=number+1
        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            rects = detector(gray, 0)  # 人脸检测

            options = {}

            options_str=""
            if smoke==0 : options_str = options_str + "smoke,"
            if cellphone==0 : options_str = options_str + "cellphone,"
            if not_buckling_up==0 : options_str = options_str + "not_buckling_up,"
            if both_hands_leaving_wheel==0 : options_str = options_str + "both_hands_leaving_wheel,"
            if not_facing_front==0 : options_str = options_str + "not_facing_front,"
            options["type"] = options_str
            options["wheel_location"] = wheel_location

            if number%timeF==0:
                print(1)
                image=cv.imencode('.jpg', frame)[1]
                a=client.driverBehavior(image,options)
                data=json.dumps(a)
                data_dict=json.loads(data)
                print("*************")
                data2=data_dict["person_info"]
                data3=data2[0]
                data4=data3['attributes']
                data5=data4['smoke']
                smoke_score=data5['score']
                data6=data4['cellphone']
                phone_score=data6['score']
                if(smoke_score>0.48):
                    cv.putText(frame, "smoking", (100, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("正在抽烟")
                if(phone_score>0.69):
                     cv.putText(frame, "calling", (200, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     print("正在打电话")
                print("*************")
                print(number)    

            for rect in rects:
                shape = predictor(gray, rect)
                points = np.zeros((68, 2), dtype=int)
                for i in range(68):
                    points[i] = (shape.part(i).x, shape.part(i).y)

                # 获取眼睛特征点
                Lefteye = points[LeftEye_Start: LeftEye_End + 1]
                Righteye = points[RightEye_Start: RightEye_End + 1]
                Mouth=points[Mouth_Start: Mouth_End + 1]

                # 计算眼睛横纵比
                Lefteye_Ratio = calculate_Ratio(Lefteye)
                Righteye_Ratio = calculate_Ratio(Righteye)
                mean_Ratio = (Lefteye_Ratio + Righteye_Ratio)/2  # 计算两眼平均比例

                #计算嘴部横纵比
                Mouth_Ratio=mouthRatio(Mouth)
                #print(Mouth_Ratio)

                # 计算凸包
                left_eye_hull = cv.convexHull(Lefteye)
                right_eye_hull = cv.convexHull(Righteye)
                mouth_hull=cv.convexHull(Mouth)
                # 绘制轮廓
                cv.drawContours(frame, [left_eye_hull], -1, [0, 255, 0], 1)
                cv.drawContours(frame, [right_eye_hull], -1, [0, 255, 0], 1)
                cv.drawContours(frame, [mouth_hull], -1, [0, 255, 0], 1)

                # 眨眼判断
                if mean_Ratio<Radio:
                    frame_counter+=1
                    #print("困了")
                    #print(frame_counter)
                    if frame_counter>=Low_radio_constant:
                        # 发出警报
                       # print("闭眼")
                        eye=1
                        if not alarm:
                            alarm = True
                        cv.putText(frame, "eye closing", (40,70),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    #print(123456)
                    eye=0
                    alarm=False
                    frame_counter=0
                
                #打哈欠判断
                if Mouth_Ratio<Mouth_Moude_Radio:
                    frame_counter_mouth+=1
                    if frame_counter_mouth>=Low_radio_constant_mouth:
                        # 发出警报
                        mouth=1
                        if not mouth_alarm:
                            mouth_alarm = True
                        cv.putText(frame, "yawning", (40, 100),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    mouth=0
                    mouth_alarm = False
                    frame_counter_mouth=0

                # 显示结果
                if mouth==1 and eye==1:
                    cv.putText(frame, "serious", (40, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif mouth==1 or eye==1:
                    cv.putText(frame, "middle", (40, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv.putText(frame, "normal", (40, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(frame, "Eye Ratio{:.2f}".format(mean_Ratio), (200, 70),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)
                cv.putText(frame, "Mouth Ratio{:.2f}".format(Mouth_Ratio), (200, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)
            #界面显示  
            cv.imshow("test", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    number=1
    main()
