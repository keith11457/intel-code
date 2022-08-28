from openvino.inference_engine import IECore
import cv2 as cv
import numpy as np
import time

def human_pose_demo():
    ie=IECore()
    for device in ie.available_devices:
        print(device)
    model_xml="D:/openvinofile/human-pose-estimation-0001.xml"
    model_bin="D:/openvinofile/human-pose-estimation-0001.bin"
    net=ie.read_network(model=model_xml,weights=model_bin)
    input_blob=next(iter(net.input_info))
    output_blob=next(iter(net.outputs))
    n,c,h,w=net.input_info[input_blob].input_data.shape
    print(n,c,h,w)
    cap=cv.VideoCapture("D:/openvino/mul_detect/daishu2.mp4")
    exec_net=ie.load_network(network=net,device_name="CPU")
    while True:
        ret,frame=cap.read()
        if ret is not True:
            break
        image=cv.resize(frame,(w,h)).transpose(2,0,1)
        print(image.shape)
        res=exec_net.infer(inputs={input_blob:[image]})
        res=res[out_blob]
        # 根据状态检查
        # res = exec_net.requests[curr_request_id].output_blobs[out_blob].buffer
        # ih, iw, ic = frame.shape
        # for obj in res[0][0]:
        #     if obj[2] > 0.25:
        #         index = int(obj[1])-1
        #         xmin = int(obj[3] * iw)
        #         ymin = int(obj[4] * ih)
        #         xmax = int(obj[5] * iw)
        #         ymax = int(obj[6] * ih)
        #         cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
        #         cv.putText(frame, labels[index] + str(obj[2]), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 8)
        print(output_blob)
        # 显示
        cv.imshow("SSD Object Detection Async", frame)
        c = cv.waitKey(1)


if __name__ == "__main__":
    human_pose_demo()