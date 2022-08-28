import numpy as np
import cv2 as cv
import dlib
from scipy.spatial import distance

# 调用人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/openvinofile/shape_predictor_68_face_landmarks.dat")

# 设定人眼标定点
LeftEye_Start = 36
LeftEye_End = 41
RightEye_Start = 42
RightEye_End = 47

Radio = 0.25  # 横纵比阈值
Low_radio_constant = 30  # 意味着连续多少帧横纵比小于Radio小于阈值时，判断疲劳


def calculate_Ratio(eye):
    """
    计算眼睛横纵比
    """
    d1 = distance.euclidean(eye[1], eye[5])
    d2 = distance.euclidean(eye[2], eye[4])
    d3 = distance.euclidean(eye[0], eye[3])
    ratio = (d1 + d2) / (2 * d3)
    return ratio


def main():
    """
    主函数
    """
    alarm = False  # 初始化警报
    frame_counter = 0  # 连续帧计数

    cap = cv.VideoCapture(0)  # 0摄像头摄像
    while cap.isOpened():
        ret, frame = cap.read()  # 读取每一帧
        frame = cv.flip(frame, 1)
        if ret:
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

                # 计算眼睛横纵比
                Lefteye_Ratio = calculate_Ratio(Lefteye)
                Righteye_Ratio = calculate_Ratio(Righteye)
                mean_Ratio = (Lefteye_Ratio + Righteye_Ratio) / 2  # 计算两眼平均比例

                # 计算凸包
                left_eye_hull = cv.convexHull(Lefteye)
                right_eye_hull = cv.convexHull(Righteye)

                # 绘制轮廓
                cv.drawContours(frame, [left_eye_hull], -1, [0, 255, 0], 1)
                cv.drawContours(frame, [right_eye_hull], -1, [0, 255, 0], 1)

                # 眨眼判断
                if mean_Ratio < Radio:
                    frame_counter += 1
                    if frame_counter >= Low_radio_constant:
                        # 发出警报
                        if not alarm:
                            alarm = True
                        cv.putText(frame, "sleeping", (10, 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    alarm = False
                    frame_counter = 0

                # 显示结果
                cv.putText(frame, "Ratio{:.2f}".format(mean_Ratio), (300, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)

            cv.imshow("test", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

