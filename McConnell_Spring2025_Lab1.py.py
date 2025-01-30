# 3.2 NumPy
#
# (1) Why is numpy faster than using for loops in Python for operations such as matrix multiplications?
#
# Python operations often take more time due to the large amount of overhead required for interpreting 
# its functions and objects. C code, however, is much more efficient. NumPy allows us to code out easy
# element by element operations that use pre-compiled C code to carry out any operations.
#
# (2) What is the data type used in numpy which makes such operations feasible? Also name a few
# differences between this data type and its similar counterpart in Python.
#
# NumPy uses an object called a ndarray, which contains a n-dimensional array of a homogeneous data type.
# Unlike python lists, these arrays are created at a fixed size. Additionally, all of the elements must
# be of the same type in a NumPy array (not the case for a python list) so that they all take up the
# same amount of memory.
# 
# (3) Go through the documentation and create a numpy array with the elements [1,2,3,4]
import numpy as np
a = np.array([1, 2, 3, 4])
print(a)

# (4) Use np.ones, np.zeros to create an array of 1’s with dimension 3x4 and an array of 0’s with
# dimension 4x3.
b = np.ones(shape=(3,4))
c = np.zeros(shape=(4,3))
print(b)
print(c)

# (5) Create a 2x3 matrix A and a 3x4 matrix B and perform a matrix multiplication using numpy.
A = np.arange(6).reshape(2, 3)
B = np.arange(12).reshape(3, 4)
C = np.matmul(A, B)
print(A)
print(B)
print(C)

# Find the eigenvalues and eigenvectors of the matrix given below?
d = np.array([[3, 1], [1, 2]])
eigenval, eigenvec = np.linalg.eig(d)
print(eigenval)
print(eigenvec)

# 3.3. Matplotlib
#
# (1) Create a line plot of the sine function over the interval [0, 2π] using Matplotlib.
import matplotlib.pyplot as plt
x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)

plt.figure(1)
plt.plot(x, y)

# (2) Add labels to the axes in a Matplotlib plot.
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# (3) Plot the 3d graph of the function given below using Matplotlib.
from mpl_toolkits import mplot3d
x = np.arange(-2*np.pi, 2*np.pi, 0.1)
y = np.arange(-2*np.pi, 2*np.pi, 0.1)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

plt.figure(2)
ax = plt.axes(projection ='3d')
ax.plot_surface(x, y, z)
plt.show()
# 3.4 SciPy
#
# (1) Solve the linear system of equations given below using SciPy.
import scipy
linsys = scipy.linalg.solve(np.array([[3, 1], [1,2]]), np.array([[9], [8]]))
print(linsys)

# (2) Find the minimum of the function given below using SciPy's optimization module.
def f(x):
    return x**2 + 2*x

res = scipy.optimize.minimize_scalar(f)
print(res)

# (3) Perform the Fourier transformation of the function given below using SciPy. Plot the frequency
# response using matplotlib.
from scipy.fft import fft, fftfreq
N = 1000
t = np.linspace(0, 1, N)
func = np.sin(100*np.pi*t) + 0.5*np.sin(160*np.pi*t)

yfft = fft(func)
xfft = fftfreq(N, 1) * N
plt.figure(3)
plt.plot(xfft, 2.0/N * np.abs(yfft))
plt.grid()
plt.show()

# 3.5 OpenCV
#
# (1) Read an image and convert it to grayscale using OpenCV.
import cv2

img = cv2.imread('capybara.jpg')
cv2.imshow('Capybara', img)
cv2.waitKey(0)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Capybara', gray_img)
cv2.waitKey(0)

# (2) Perform edge detection on the image using OpenCV. 
edge_img = cv2.Canny(gray_img, 150, 200)
cv2.imshow('Edge Detected Capybara', edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# (3) Use a Haar cascade classifier to implement face detection using OpenCV on an image which
# contains faces.
img_face = cv2.imread('me.jpg')
gray_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
import os
print(os.getcwd())
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detect = haar_cascade.detectMultiScale(gray_face, 1.1, 9)

for (x, y, w, h) in detect: 
    cv2.rectangle(img_face, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Face Detection', img_face)
cv2.waitKey(0)

# Homework
# A Haar cascade involves scanning an image and categorizing each area into a different type of Haar feature,
# which is a combination of white and black rectangles. If a region of the image conatins a grouping of
# Haar features that resembles a face, it is sent to more stages (a cascade) to verify that it is indeed a
# face. A feature or collection of them is only sent to the next stage if its error is below a certain
# threshold so that the algorithm isn't accounting for an unnecessarily large amount of features. Additionally,
# features will be categorized based larger groups of pixels rather than a singular one to save time.