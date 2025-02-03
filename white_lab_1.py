
#3.2
#1 Why is numpy faster than using for loops in Python for operations such as matrix multiplications?
#
# numpy is specialized to work with machine data types specifically so the individual integers of the 
#do not need to be loaded back and forth with every manipulation.

#2 hat is the data type used in numpy which makes such operations feasible? Also name a few
#differences between this data type and its similar counterpart in Python


import numpy as np 
#3
a = np.array([1, 2, 3, 4])
#4
b = np.ones((3,4))
c = np.zeros((4,3))
#5 
A = np.array([1, 2, 3],
             [4, 5, 6])
B = np.array([1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 0, 1, 2])
C = A @ B
#6
D = ([3, 1],
     [1, 2])
eigenvaues, eigenvectors = np.linalg.eig(D)

#3.3
#1
import matplotlib.pyplot as plt
import math
x = np.arange(0,(2*pi))
fig, ax = plt.subplots()
ax.plot([0,2*pi], [0, pi])
fig, ax = plt.subplots(figsize=(0, 2*pi))
t = np.sin(x)
line, = ax.plot(t, lw=2)
#2
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
#3
result = math.sqrt((x^2) + (y^2))
z = np.sin(result)
line, = ax.plot(t, z, lw=2)

#3.4
#1
from scipy import linalg
A = np.array([3,1],
             [1,2])
b = np.array([9]
             [8])
C=linalg.inv(A).dot(b)
#2
from scipy.optimize import minimize_scalar

def func(x):
    return x**2 + 2*x

result = minimize_scalar(func)

print("Minimum value:", result.fun)
print("x value at minimum:", result.x)
#3
from scipy.fft import fft, fftfreq
import numpy as np
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(100*np.pi*x) + 0.5*np.sin(160*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

#3.5
# for this entire section, I had no clue where to begin on the 
# tutorials as they were not clearly written for this assignement.
# I therefore used the first things that came up as solutions 
# in google. I will keep reading but the scale of this assignment
# was overwhelming for my first time coding in python
#1
# import opencv
import cv2

# Load the input image
image = cv2.imread('C:\\Documents\\full_path\\tomatoes.jpg')
cv2.imshow('Original', image)
cv2.waitKey(0)

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)  

# Window shown waits for any key pressing event
cv2.destroyAllWindows()

#2
import cv2
 
# Read the original image
img = cv2.imread('test.jpg') 
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
 
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)
 
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
 
cv2.destroyAllWindows()

#3
# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the input image
image = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 
                                     scaleFactor=1.1, 
                                     minNeighbors=5, 
                                     minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the output image
cv2.imshow('Faces Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#HW
#Haar cascades detect faces through a number of cocepts
# 1 They disect the face image into features that it can subdivide
# for recognition like eyes, mouth, nose etc.
# 2, they set up positive and negative indications in the image
# and eliminate the negatives, thus rejecting negative, and 
# allowing efficient use of memory.
# 3 THey image comparisons are put into weak and strong recognition
# categories, and are cascaded into likelyhood. 
