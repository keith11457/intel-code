from unittest import result
from aip import AipBodyAnalysis
import json
import cv2 as cv
from numpy import var
from sqlalchemy import true
from threading import Thread
from playsound import playsound
from pygame import mixer
""" 你的 APPID AK SK """
#可修改的数据
""" 你的 APPID AK SK """
APP_ID = '26645471'
API_KEY = 'O5Ua1M3BfUmWjwVhkfn8Lf5b'
SECRET_KEY = 'mlsbqfwaub0DBclt0tsXETPPA4VhyqQ6'

""" 图片地址 """
image_a="diver.jpeg"
capture = cv.VideoCapture(2)#0为默认摄像头

def camera():
    while True:
        # 获得图片
        ret, frame = capture.read()
        #cv2.imshow("窗口名称", "窗口显示的图像)
        # 显示图片
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

def warnaction():

    Thread(target=camera).start()  # 引入线程防止在识别的时候卡死

    """ 需要的参数（0则为需要，1则为不需要） """
    smoke=0 #吸烟
    cellphone=0 #打手机
    not_buckling_up=1 #未系安全带
    both_hands_leaving_wheel=1 #双手离开方向盘
    not_facing_front=1 #视角未看前方
    """ 可按实际情况修改 """
    """ "1"，表示左舵车 "0"，表示右舵车 """
    wheel_location=1
    client=AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY) #生成key
    number=1
    timee=0

    while(1):
        ret,frame = capture.read()
        cv.imshow('frame', frame)
        #image=get_file_content(frame)
        image = cv.imencode('.jpg', frame)[1]
        """ 调用驾驶行为分析 """
        client.driverBehavior(image)

        """ 如果有可选参数 """
        options = {}
        timee=timee+1
        options_str=""
        if smoke==0 : options_str = options_str + "smoke,"
        if cellphone==0 : options_str = options_str + "cellphone,"
        if not_buckling_up==0 : options_str = options_str + "not_buckling_up,"
        if both_hands_leaving_wheel==0 : options_str = options_str + "both_hands_leaving_wheel,"
        if not_facing_front==0 : options_str = options_str + "not_facing_front,"
        options["type"] = options_str
        options["wheel_location"] = wheel_location
        print("*************")
        print(options)

        """ 带参数调用驾驶行为分析 """
        a=client.driverBehavior(image, options)
        #print(json.dumps(a,indent=3)) #格式化输出
        data=json.dumps(a)
        data_dict=json.loads(data)
        print(number)
        #result=json.loads()
        data2=data_dict["person_info"]
        data3=data2[0]
        data4=data3['attributes']
        data5=data4['smoke']
        smoke_score=data5['score']
        data6=data4['cellphone']
        phone_score=data6['score']
        if(smoke_score>0.48 and timee>1):
            print("正在抽烟")
            mixer.init()  # 初始化混音器模块
            mixer.music.load(r'C:\Users\Administrator\Desktop\吸烟.mp3')  # 载入待播放音乐文件
            mixer.music.play()  # 开始播放音乐流
            timee=0
        if(phone_score>0.69 and timee>1):
            print("正在打电话")
            mixer.init()  # 初始化混音器模块
            mixer.music.load(r'C:\Users\Administrator\Desktop\电话.mp3')  # 载入待播放音乐文件
            mixer.music.play()  # 开始播放音乐流
            timee=0
        if cv.waitKey(1) & 0xFF == ord('q'):
                 break

    capture.release()
    cv.destroyAllWindows()

if __name__=="__main__":
    warnaction()