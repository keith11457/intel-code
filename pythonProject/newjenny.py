import serial #导入串口通信库
from time import sleep

ser = serial.Serial()

ser.port = 'COM9'
ser.baudrate = 9600
ser.stopbits = 1
ser.bytesize = 8
ser.parity = 'N'
print('start')
while (ser.isOpen() == 0):
        print('打开串口失败')
        ser.open()
ser.write(b"2")
while 1:
        alcohol_num = ser.readline().decode()
        print(alcohol_num)
        if alcohol_num=="8\r\n":
                    ser.write(b"2")
                    sleep(1)
                    print('ok')
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# import serial
# import time
# serialPort = "COM9"  # 串口
# baudRate = 9600  # 波特率
# ser = serial.Serial(serialPort, baudRate, timeout=0.5)
# print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))
#
# demo1=b"0"#将0转换为ASCII码方便发送
# demo2=b"1"#同理
# while 1:
#         num = ser.readline().decode()
#         print(num)
    # if alcohol_num=='1':
    #                     ser.write(b"2")
    #                     time.sleep(1)
    #                     print("ok")
