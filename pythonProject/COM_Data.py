import serial#导入串口通信库
from time import sleep

ser = serial.Serial()

def port_open_recv():#对串口的参数进行配置
    ser.port='com7'
    ser.baudrate=9600
    ser.bytesize=8
    ser.stopbits=1
    ser.parity="N"#奇偶校验位
    ser.open()
    if(ser.isOpen()):
        print("串口打开成功！")
    else:
        print("串口打开失败！")
#isOpen()函数来查看串口的开闭状态



def port_close():
    ser.close()
    if(ser.isOpen()):
        print("串口关闭失败！")
    else:
        print("串口关闭成功！")

def send(send_data):
    if(ser.isOpen()):
        ser.write(send_data.encode('utf-8'))#编码
        print("发送成功",send_data)
    else:
        print("发送失败！")

if __name__ == '__main__':
    ser.port='COM7'
    ser.baudrate=9600
    ser.stopbits=1
    ser.bytesize=8
    ser.parity='N'
    ser.open()
    while (ser.isOpen()==0):
        print('打开串口失败')
    print('打开串口成功')
    while True:
        print(ser.readline())

