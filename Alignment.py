import numpy as np
import matplotlib.pyplot as plt
import imutils
import math
import cv2
import sys

#display image
def print_image(name, image):
	cv2.imshow(name, image)
	cv2.waitKey()
	# cv2.destroyAllWindows()

def BGR_RGB(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image

def resize(image):
	dim = (900, 500)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def gaussian_blur(image, kernel_size):
	gray = cv2.GaussianBlur(image, (kernel_size, kernel_size),20)
	return gray

def rotation(image, angle):
	rows,cols = image.shape[:2]
	# make the angle negative for opposite direction
	M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
	rotated = cv2.warpAffine(image, M, (cols, rows))
	return rotated

def corner_bound_rotation(image, angle):
	rotated = imutils.rotate_bound(image, -angle)
	return rotated

def CannyEdge(image):
	# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# edges = cv2.Canny(image, 30,150,apertureSize = 3) #works for most cases
	edges = cv2.Canny(image, 5, 200,apertureSize = 5)
	return edges

def HoughTransform(image, edges):
	lines = cv2.HoughLines(edges,5, np.pi/180,100)
	print(len(lines[0]))
	angles=[]
	for rho,theta in lines[0]:
		print(rho,theta)
		angle_new = theta/np.pi
		angle_new = (angle_new*180) - 120
		angle_new = (angle_new*np.pi)/180
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 - 170*(-b))
		y1 = int(y0 - 170*(a))
		x2 = int(x0 - 850*(-b))
		y2 = int(y0 - 850*(a))
		cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)	
		angle = math.degrees(math.atan2(y2-y1, x2-x1))
		angles.append(angle)
		median_angle = np.median(angles)	

		rect = np.zeros((4, 2), dtype = "float32")
		a = np.cos(angle_new)
		b = np.sin(angle_new)
		x0_ = a*rho
		y0_ = b*rho
		x1_ = int(x1)
		y1_ = int(y1)
		x2_ = int(x0_ + 400*(-b))
		y2_ = int(y0_ + 400*(a))
		# cv2.line(image,(x1_,y1_),(x2_,y2_),(0,0,255),2)	
		# image[x1,y1]=(0,255,255)
		# image[x1,y1] = (0,0,255)
		# image[x2,y2] = (0,0,255)
		# image[x2_,y2_] = (0,0,255)

		# pts1 = np.float32([[x1,y1],[x2,y2],[x1_,y1_],[x2_,y2_]])
		# pts2 = np.float32([[499,0],[499,899],[499,0],[0,0]])
		# M = cv2.getPerspectiveTransform(pts1,pts2)
		# dst = cv2.warpPerspective(resized,M,(900,500))
		# plt.subplot(121),plt.imshow(image),plt.title('Input')
		# plt.subplot(122),plt.imshow(dst),plt.title('Output')
		# plt.show()
		
		# # pts2 = np.float32([[10,100],[200,50],[100,250]])

		# cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
		# rho = rho - 300
		# x0_ = a*rho
		# y0_ = b*rho
		# x1_ = int(x0_ + 1000*(-b))
		# y1_ = int(y0_ + 1000*(a))
		# x2_ = int(x0_ - 1000*(-b))
		# y2_ = int(y0_ - 1000*(a))

		# cv2.line(image,(x1_,y1_),(x2_,y2_),(0,0,255),2)


		# pts1 = np.float32([[x1,y1],[x2,y2],[x2_,y2_]])
		# pts2 = np.float32([[0,900],[500,900],[500,0]])
		# M = cv2.getAffineTransform(pts1,pts2)
		# dst = cv2.warpAffine(image,M,(500,900))

		# dst = cv2.resize(dst, (900,500), interpolation = cv2.INTER_AREA)

		# plt.show()

		# print(pts1)
		# print(pts2)
		# M = cv2.getPerspectiveTransform(pts1,pts2)
		# dst = cv2.warpPerspective(image,M,(900,500))


	# plt.imshow(dst),plt.title('Output')
	print(median_angle)
	return image, lines, median_angle



# def hough_angle(image, lines):
# 	# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# 	angles = []
# 	for rho,theta in lines[0]:
# 		a = np.cos(theta)
# 		b = np.sin(theta)
# 		x0 = a*rho
# 		y0 = b*rho
# 		x1 = int(x0 + 1000*(-b))
# 		y1 = int(y0 + 1000*(a))
# 		x2 = int(x0 - 1000*(-b))
# 		y2 = int(y0 - 1000*(a))
# 	    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
#     	angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#     	angles.append(angle)
# 	median_angle = np.median(angles)
# 	return median_angle

image_gray = cv2.imread(sys.argv[1])
# print_image("RBG", image_gray)

resized = resize(image_gray)
print_image("resized", resized)


# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
# M = cv2.getPerspectiveTransform(pts1,pts2)
# dst = cv2.warpPerspective(resized,M,(300,300))
# plt.subplot(121),plt.imshow(resized),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()

blurred = gaussian_blur(resized, 5)
# print_image("blurred", blurred)

canny_edges = CannyEdge(blurred)
print_image("canny_edges", canny_edges)

# hough_image, lines, angle = HoughTransform(resized, canny_edges)
# # print(lines[0])


# angle_new = angle/np.pi
# angle_new = (angle_new*180) - 90
# angle_new = (angle_new*np.pi)/180



# hough_angle = hough_angle(hough_image, lines)
# print(hough_angle)

# Aligned_image = rotation(resized, angle)
# print_image("Aligned_image", Aligned_image)
# resized = resize(Aligned_image)

# blurred = gaussian_blur(resized,3)
# print_image("blurred", blurred)

# canny_edges = CannyEdge(blurred)
# print_image("canny_edges", canny_edges)

hough_image, lines, angle = HoughTransform(resized, canny_edges)
print_image("hough", hough_image)
# print(lines[0])
Aligned_image = rotation(resized, angle)
print_image("Aligned_image", Aligned_image)

blurred = gaussian_blur(Aligned_image, 3)
# print_image("blurred", blurred)

canny_edges = CannyEdge(blurred)
# print_image("canny_edges", canny_edges)

hough_image, lines, angle = HoughTransform(Aligned_image, canny_edges)
print(angle)
