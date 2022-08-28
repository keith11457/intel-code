# -*- coding:utf-8 -*-
from openvino.inference_engine import IENetwork, IECore
import cv2
import logging
import os
import sys
from time import time


logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def main():
    # ------------ 1. 模型导入 ------
    model_xml = "D:/openvinofile/vehicle-license-plate-detection-barrier-0106.xml"
    model_bin = "D:/openvinofile/vehicle-license-plate-detection-barrier-0106.bin"
    log.info("创建推理引擎...")
    ie = IECore()
    # ------------ 2. 读取模型优化器生成的IR（.xml和.bin文件） ------
    net = ie.read_network(model=model_xml, weights=model_bin)
    # ---------------------------- 3. 准备输入 ------------
    log.info("准备输入")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))  # 后面添加
    print(out_blob)
    #  默认批量是1
    net.batch_size = 1
    # 读取并预处理输入图像
    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n,c,h,w)
    cap = cv2.VideoCapture("D:/openvinofile/mul_detect/plate.mp4")  # car_test1.mp4 car_1.bmp
    # --------------------------- 4. 将模型加载到插件 ---------------------------
    log.info("将模型加载到插件")
    exec_net = ie.load_network(network=net, device_name="CPU")
    # --------------------------- 5. 开始推断... --------------------------
    log.info("开始推断...")
    while True:
        ret, in_frame = cap.read()
        # frame = cv.GaussianBlur(frame, (3, 3), 0, 0)
        if ret is not True:
            break
        # 将输入框调整为网络大小
        frame=cv2.resize(in_frame,(w,h))
        frame = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        print(frame.shape)
        # 开始推断
        start_time = time()
        res=exec_net.infer(inputs={input_blob:[frame]})
        det_time = time() - start_time  # 推理时间
        print("infer time(ms)：%.3f" % (det_time * 1000))#输出推理时间
        initial_w,initial_h,initial_c=in_frame.shape
        res=res[out_blob]
        s=0
        for obj in res[0][0]:
            if obj[2] > 0:  # obj[2] : conf
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                class_id = int(obj[1])

                # color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                cv2.rectangle(in_frame, (xmin, ymin), (xmax, ymax), (24, 225, 30), 2)
                cv2.putText(in_frame, ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (210, 230, 20), 1)

        start_time = time()
        image = frame.transpose(2, 0, 1)
        cv2.imshow("DetectionResults", in_frame)
        cv2.waitKey(1)
        render_time = time() - start_time

    cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()