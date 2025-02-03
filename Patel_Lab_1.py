import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.fftpack import fft, fftfreq
import cv2 as cv
import sys

# 3.2 NumPy

# 1) Why is numpy faster than using for loops in Python for operations such as matrix multiplications?

# Numpy is faster for operations because Numpy uses vectorization which is absense of explicit looping/indexing
# as these things take place in optimzied, pre-compiled C code.

# 2) What is the data type used in numpy which makes such operations feasible? Also name a few differences between this data type 
# and its similar counterpart in Python

# In NumPy, the data type that enables operations on arrays is called "ndarray", that is Numpy Array, which allows for homogenous 
# data storage, meaning all elements within an array must be of the same data type (like integer, float, boolean), unlike Python
# lists where elements can be of different types.

# 3)
a = np.array([1, 2, 3, 4])
print("Example of numpy array:", a)

# 4)
ones_array = np.ones((3,4))
zeroes_array = np.zeros((4,3))
print("Array of 1’s with dimension 3x4:")
print(ones_array)
print("Array of 0’s with dimension 4x3:")
print(zeroes_array)

# 5)
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
result = np.dot(A, B) 
print("Example of Matrix Multiplication:")
print(result) 

# 6)
A = np.array([[3, 1], [1, 2]])
D,V = np.linalg.eig(A) 
print("Eigenvalues:", D) 
print("Eigenvectors:\n", V)

# 3.3 Matplotlib

# 1 and 2)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
# Plot the sine function
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Function')
plt.show()

#3)
# Define the function z = sin(x^2 + y^2)
def func(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

# Create a grid of x and y values
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

# Add labels and a color bar
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.title('3D plot of z = sin(x^2 + y^2)')
plt.show()

# 3.4 SciPy

# 1)
A = ([3, 1],[1, 2])
B = ([9], [8])
x = la.solve(A,B)
print("The solution for x and y is:", x)

# 2)

# Define the function y = x^2 + 2x
def function(x):
    return (x**2 + 2*x)
# Minimize the function
result = minimize(function,1).x
print("The minimum of the function is:", result)

# 3)

# Define the function
x = np.linspace(0, 1, 1000, endpoint=False)  # 1000 points in the interval [0, 1)
f_x = np.sin(100 * np.pi * x) + 0.5 * np.sin(160 * np.pi * x)

# Perform Fourier Transformation
fft_result = fft(f_x)
frequencies = fftfreq(len(x), d=(x[1] - x[0]))  # Frequency bins

# Take only positive frequencies for plotting
positive_freqs = frequencies[frequencies >= 0]
positive_fft = np.abs(fft_result[frequencies >= 0])

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, positive_fft, label="Frequency Response")
plt.title("Frequency Response of the Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

# 3.5 Open CV

#1)
#Read in Image
img = cv.imread(cv.samples.findFile("image.png"))

if img is None:
    sys.exit("Could not read the image.")
 
#Grayscale the image
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Filter and Edge Detection
blurred = cv.GaussianBlur(src=gray_image, ksize=(3, 5), sigmaX=0.5)  
edges = cv.Canny(blurred, 70, 135) 

#Face Detection
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'Haarcascade_frontalface_default.xml') 
faces_rect = haar_cascade.detectMultiScale(gray_image, 1.1, 9) 

for (x, y, w, h) in faces_rect: 
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 

# Convert images from BGR to RGB for correct display in matplotlib
image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray_image_rgb = cv.cvtColor(gray_image, cv.COLOR_BGR2RGB)
edges_rgb = cv.cvtColor(edges, cv.COLOR_BGR2RGB)

# Plot the images using matplotlib
plt.figure(figsize=(7, 5))

# Plot grayscale image
plt.subplot(1, 3, 1)
plt.imshow(gray_image_rgb)
plt.title('Grayscale Image')
plt.axis('off')

# Plot edge detection result
plt.subplot(1, 3, 2)
plt.imshow(edges_rgb, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Plot face detection result
plt.subplot(1, 3, 3)
plt.imshow(image_rgb)
plt.title('Face Detection')
plt.axis('off')

# Show all images
plt.show()

