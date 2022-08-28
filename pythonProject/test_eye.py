"""
图片的人脸标定
"""
# coding: utf-8
import cv2
import numpy as np
import dlib
from scipy.spatial import distance

# draw_all_landmarks_id
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('/home/tanhui/notebook/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('D:/openvinofile/shape_predictor_68_face_landmarks.dat')

img = cv2.imread('D:/PycharmProjects/pythonProject2/0.jpg')
print(img.shape)
dets = detector(img, 0)  # dlib人脸检测
print(dets[0])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/openvinofile/shape_predictor_68_face_landmarks.dat")

# 设定人眼标定点
LeftEye_Start = 36
LeftEye_End = 41
RightEye_Start = 42
RightEye_End = 47

Radio = 0.2  # 横纵比阈值
Low_radio_constant = 2  # 当Radio小于阈值时，接连多少帧一定发生眨眼动作


def calculate_Ratio(eye):
    """
    计算眼睛横纵比
    """
    d1 = distance.euclidean(eye[1], eye[5])
    d2 = distance.euclidean(eye[2], eye[4])
    d3 = distance.euclidean(eye[0], eye[3])
    d4 = (d1 + d2)/2
    ratio = d4 / d3
    return ratio

def main_test_eye(self):
    for i, d in enumerate(dets):
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
        shape = predictor(img, d)  # dlib人脸特征点检测
        points = np.zeros((68, 2), dtype=int)
        for k in range(0, 68):  # 68个特征点
            cv2.circle(img,(shape.part(k).x, shape.part(k).y), 2, (0, 0, 255), -1)  #-1表示填充
            points[k] = (shape.part(k).x, shape.part(k).y)
        Lefteye = points[LeftEye_Start: LeftEye_End + 1]
        Righteye = points[RightEye_Start: RightEye_End + 1]
        Lefteye_Ratio = calculate_Ratio(Lefteye)
        Righteye_Ratio = calculate_Ratio(Righteye)
        mean_Ratio = (Lefteye_Ratio + Righteye_Ratio) / 2  # 计算两眼平均比例
        print(mean_Ratio)

            #cv2.putText(img,'%d' % k,(shape.part(k).x,shape.part(k).y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1) #标记点号
            #print (shape.part(k).x,shape.part(k).y)
    cv2.imwrite('face_landmarks.jpg', img)
    # print('success')

    cv2.imshow('landmarks', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
