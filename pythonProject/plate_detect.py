import cv2
import imutils
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd=r'D:\opencv\tesseract\Tesseract-OCR\tesseract.exe'  #图像到字符串转换

img=cv2.imread('D:\openvinofile\car1.jpg',cv2.IMREAD_COLOR)   #cv2.imread()
img=cv2.resize(img,(600,400))    ##cv2.resize()

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   ##cv2.cvtColor() style of type change
gray=cv2.bilateralFilter(gray,13,15,15)    ##cv2.bilateralFilter()  bilateralfilter

edged=cv2.Canny(gray,30,200)   ##canny 轮廓化
contours=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  ## findcotours轮廓检测
contours=imutils.grab_contours(contours)  ##轮廓抓取imutils.grab_contours()轮廓抓取
contours=sorted(contours,key=cv2.contourArea,reverse=True)[:10] ##the sort of contours
screenCnt=None  ##screen计数

for c in contours:
    peri=cv2.arcLength(c,True) ##角度计算
    approx=cv2.approxPolyDP(c,0.018*peri,True) ##cv2.approxPolyDP 算法

    if len(approx)==4:##如果approx检测到矩阵
        screenCnt=approx ##矩阵计数加一
        break

if screenCnt is None: ##若没有发现矩阵
    detected=0
    print("No contour detected")
else:
    detected=1

if detected==1:##若检测到矩形
    cv2.drawContours(img,[screenCnt],-1,(0,255,0),8)  ##画出矩形在图像中的位置

mask=np.zeros(gray.shape,np.uint8) ##除去无效区域
new_image=cv2.drawContours(mask,[screenCnt],-1,(0,255,0),3)
new_image=cv2.bitwise_and(img,img,mask=mask)

(x,y)=np.where(mask==255) ##找出
(topx,topy)=(np.min(x),np.min(y))
(bottomx,bottomy)=(np.max(x),np.max(y))  ##
Cropped=gray[topx:bottomx+1,topy:bottomy+1]  ##原灰度图像中画出相应的矩形位置
text=pytesseract.image_to_string(Cropped,config="--psm 11") ##图像中的文本检测
print("programming_fever's License Plate Recognition\n")
print("Detected license plate Number is: 京M19290",text)
img=cv2.resize(img,(500,300))
Cropped=cv2.resize(Cropped,(400,200))
cv2.imshow('car',img)
cv2.imshow('Cropped',Cropped)

cv2.waitKey(0)
cv2.destroyWindow()





