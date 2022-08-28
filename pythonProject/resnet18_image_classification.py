from openvino.inference_engine import IECore
import numpy as np
import cv2 as cv

ie = IECore()
for device in ie.available_devices:
    print(device)

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

model_xml = "resnet18.xml"
model_bin = "resnet18.bin"

net = ie.read_network(model=model_xml, weights= model_bin)
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

n, c, h, w = net.input_info[input_blob].input_data.shape
print(n, c, h, w)

src = cv.imread("D:/openvinofile/messi.jpg")
image = cv.resize(src, (w, h))
image = np.float32(image) / 255.0
image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
image = image.transpose(2, 0, 1)

exec_net = ie.load_network(network=net, device_name="CPU")
res = exec_net.infer(inputs={input_blob:[image]})

res = res[out_blob]
print(res.shape)
label_index = np.argmax(res, 1)[0]
print(label_index, labels[label_index])
cv.putText(src, labels[label_index], (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, 8)
cv.imshow("image classification", src)
cv.waitKey(0)
