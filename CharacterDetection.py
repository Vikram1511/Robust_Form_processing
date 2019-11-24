import cv2 
import sys
import numpy as np

def resize(image):
    dim = (900,500)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized 
    
#import image 
image = cv2.imread(sys.argv[1])
image = resize(image)
#grayscale 
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
cv2.imshow('gray', gray) 
cv2.waitKey(0) 
#binary 
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,8)
cv2.imshow('second', thresh) 
cv2.waitKey(0) 
#dilation 
kernel = np.ones((1,1), np.uint8) 
dil = cv2.dilate(thresh, kernel) 
cv2.imshow('dilated', dil) 
cv2.waitKey()

cv2MajorVersion = cv2.__version__.split(".")[0] 

if (int(cv2MajorVersion) == 4):
	ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
else:
	im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
	x, y, w, h = cv2.boundingRect(ctr)
	roi = image[y:y+h, x:x+w]
	cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2)

cv2.imshow('detected characters',image) 
cv2.waitKey(0)
