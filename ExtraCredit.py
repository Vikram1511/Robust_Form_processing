import sys
import numpy as np
import cv2

#display image
def print_image(name, image):
	cv2.imshow(name, image)
	cv2.waitKey()
	cv2.destroyAllWindows()


def resize(image, width, height):
	dim = (width, height)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def gaussian_blur(image, kernel_size):
	gray = cv2.GaussianBlur(image, (kernel_size, kernel_size),0)
	return gray

kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT,(1,7))
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))

img = cv2.imread(sys.argv[1])
resized = resize(img, 900, 500)

gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,1)
print_image("thresh", thresh)

output_image_v = cv2.morphologyEx(thresh, cv2.MORPH_HITMISS, kernel_v)
print_image("output_image", output_image_v)

output_image_h = cv2.morphologyEx(thresh, cv2.MORPH_HITMISS, kernel_h)
print_image("output_image", output_image_h)

add = cv2.add(output_image_v, output_image_h)
print_image("add", add)


# Alternative approach
# import cv2 
# import sys
# import numpy as np

# def resize(image):
#     dim = (900,500)
#     resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#     return resized 

# def gaussian_blur(image, kernel_size):
# 	gray = cv2.GaussianBlur(image, (kernel_size, kernel_size),0)
# 	return gray
# #import image 
# image = cv2.imread(sys.argv[1])
# image = resize(image)
# #grayscale 

# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
# gray = gaussian_blur(gray, 3)
# cv2.imshow('gray', gray) 
# cv2.waitKey(0) 
# #binary 
# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 4)
# cv2.imshow('second', thresh) 
# cv2.waitKey(0) 
# #dilation 
# kernel = np.ones((1,1), np.uint8) 
# img_dilation = cv2.dilate(thresh, kernel, iterations=1) 
# cv2.imshow('dilated', img_dilation) 
# cv2.waitKey()

# cv2MajorVersion = cv2.__version__.split(".")[0] 
# # check for contours on thresh 
# if (int(cv2MajorVersion) == 4):
# 	ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
# else:
# 	im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #sort contours
# sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

# for i, ctr in enumerate(sorted_ctrs):
# 	x, y, w, h = cv2.boundingRect(ctr)
# 	area = w*h
# 	if (10 <area < 200 and w <20 and h <20):
# 		rect = cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 0, 0), 2)
# 		cv2.imshow('rect', rect)
	
# 	# if w > 30 and h > 20 and h< 100 and w < 100:
# 	# 	cv2.imshow("roi", roi)

# cv2.imshow('marked areas', thresh) 
# cv2.waitKey(0)
