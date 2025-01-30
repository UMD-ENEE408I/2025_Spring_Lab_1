# Eric Stearns 


# Lab 1 numpy questions
# 
# 
# Q1) Why is numpy faster than using for loops in Python for operations such 
#     as matrix multiplications?
#  
# Numpy arrays are stored as continuous blocks in memory which makes it a 
# lot easier to access their data.
# 
# 
# Q2) What is the data type used in numpy which makes such operations 
#     feasible? Also name a few differences between this data type and its 
#     similar counterpart in Python
# 
# Numpy arrays make these operations feasible. Numpy arrays are homogenous
# by default while Python lists can be homogenous or heterogenous. Numpy
# arrays also allow allow element-wise operations unlike Python lists.
# Numpy arrays also can more effectively create multidimensional lists,
# as multidimensional lists in Python are effectively 1D lists of 1D lists.
# 
# 
# Q3) Go through the documentation and create a numpy array with the 
#     elements [1,2,3,4]

import numpy as np
a = np.array([1,2,3,4])
print(a)

# Q4) Use np.ones, np.zeros to create an array of 1’s with dimension
#     3x4 and an array of 0’s with dimension 4x3.

b = np.ones((3,4))
c = np.zeros((4,3))
print(b)
print(c)

# Q5) Create a 2x3 matrix A and a 3x4 matrix B and perform a matrix 
# multiplication using numpy.

A = np.arange(6).reshape(2, 3)
print(A)
B = np.arange(12).reshape(3, 4)
print(B)
C = np.matmul(A,B)
print(C)

# Q6) Find the eigenvalues and eigenvectors of the matrix given below?

mtx = np.array([[3,1],
                [1,2]])
print(np.linalg.eig(mtx))

# Lab 1 matplotlib questions
# 
# 
# Q1) Create a line plot of the sine function over the interval [0, 2π] 
# using Matplotlib.

import matplotlib.pyplot as plt
import numpy as np

fig,ax = plt.subplots()
x = np.arange(0.0,2*np.pi,0.01)
y = np.sin(x)
ax.plot(x, y)
plt.show()

# Q2) Add labels to the axes in a Matplotlib plot.


fig,ax = plt.subplots()
x = np.arange(0.0,2*np.pi,0.01)
y = np.sin(x)
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("sinx")
plt.show()

# Q3) Plot the 3d graph of the function given below using Matplotlib.
# z = sin(sqrt(x^2+y^2))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10, 0.25)
x, y = np.meshgrid(x, y)
R = np.sqrt(x**2 + y**2)
z = np.sin(R)
surf = ax.plot_surface(x,y,z)
plt.show()

# Lab 1 scipy questions
# 
#
# Q1) Solve the linear system of equations given below using SciPy.

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

A = np.array([[3,1],
              [1,2]])
B = np.array([[9],
              [8]])
sol = np.linalg.solve(A, B)
print(sol)

# Q2) Find the minimum of the function given below using SciPy's optimization module.

from scipy.optimize import minimize_scalar

def f(x):
    return x**2+2*x

res = minimize_scalar(f)
print(res.fun)
print(res.x)

# Q3) Perform the Fourier transformation of the function given below using SciPy. 
# Plot the frequency response using matplotlib.

from scipy.fft import fft, fftfreq, fftshift
# number of signal points
N = 1000
# sample spacing
T = 1.0 / 1000.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(100*np.pi*x) + np.sin(160*np.pi*x)/2
yf = fft(y)
xf = fftfreq(N, T)
xf = fftshift(xf)
yplot = fftshift(yf)
plt.plot(xf, 1.0/N * np.abs(yplot))
plt.grid()
plt.show()


# OpenCV questions
import cv2

# Load the image
 
image_name = "anger.png"
image = cv2.imread(cv2.samples.findFile(image_name), cv2.IMREAD_COLOR)
cv2.imshow("Original Image", image)

# Wait for the user to press a key
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(gray_image, (3,3), 0) 
 
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

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=25, threshold2=150) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
 
cv2.destroyAllWindows()

image2 = cv2.imread('face_img.JPG')
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) 
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
faces_rect = haar_cascade.detectMultiScale( 
    gray_image2, scaleFactor=1.1, minNeighbors=9) 
for (x, y, w, h) in faces_rect: 
    cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) 
  
cv2.imshow('Detected faces', image2) 
cv2.waitKey(0) 

# homework question:
# Haar cascades can detect faces by detecting light and dark features in an image.
# It creates composite images of light and dark rectangles and compares them against
# what it has been told are or are not faces in the past.