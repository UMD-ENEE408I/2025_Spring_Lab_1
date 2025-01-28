# Question 1
# Numpy is faster than using for loops in Pyton for operations such as matrix multiplication because it applies multiplication to the entire array at once, instead of multiplying each individual element.

# Question 2
# The data type used in numpy which makes such operations feasible is numpy arrays. In python, the similar data type is lists. 
# Arrays in numpy, the elements in the array have to be the same type while elements in lists in python do not have to share the same type.

## numpy ##
import numpy as np
# Question 3
print(np.array([1,2,3,4]))

# Question 4
ones_array = np.ones((3,4))
zeros_array = np.zeros((4,3))
print(ones_array)
print(zeros_array)

# Question 5
matrix_a = np.array([[1,2,3], [4,5,6]])
matrix_b = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(matrix_a)
print(matrix_b)
print(np.dot(matrix_a, matrix_b))

# Question 6
arr = np.array([[3,1], [1,2]])
eigenvalues, eigenvectors = np.linalg.eig(arr)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

## matplotlib ##
# 1
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.figure(figsize=(8, 4))         
plt.plot(x, y)

# 2
plt.xlabel('x label')
plt.ylabel('y label')
plt.show()

# 3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-5, 5, 100)  
y = np.linspace(-5, 5, 100)   
x, y = np.meshgrid(x, y)      
z = np.sin(np.sqrt(x**2 + y**2))     
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)
plt.show()

## SciPy ##
import numpy as np
from scipy.linalg import solve
# 1
A = np.array([[3,1], [1,2]])
b = np.array([9, 8])

x = solve(A,b) 
print(x)

# 2
from scipy.optimize import minimize
def func(x):
    return x**2 + 2*x
x0 = 0
result = minimize(func, x0)
print("Minimum value:", result.fun)
print("Value of x at the minimum:", result.x)

# 3
from scipy.fft import fft, fftfreq
def f(x):
    return np.sin(100 * np.pi * x) + 0.5 * np.sin(160 * np.pi * x)
x = np.linspace(-1, 1, 1000)
y = f(x)
Y = fft(y)
frequencies = fftfreq(len(x), x[1] - x[0])
plt.figure()
plt.plot(frequencies, np.abs(Y))
plt.show()

## OpenCv ##
# 1 and 2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('kungfupanda.jpg', cv.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"

edges = cv.Canny(img,100,200)

plt.figure(figsize=(12, 6))

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 3
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def detect_faces(image_path):
    # Load the Haar cascade classifier
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 
                                         scaleFactor=1.1, 
                                         minNeighbors=5, 
                                         minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the image
    cv.imshow('Faces Detected', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

detect_faces('face.jpg')

# 4
# Haar cascade detects faces by sliding a window across an image, extracting features from each window using Haar wavelets, and classifying each window as either containing a face or not based on a trained classifier that differentiates between positive (face) and negative (non-face) image samples.
# Bascially, it identifies facial features like eyes, nose, and mouth by looking for specific pixel intensity differences within the window, allowing it to detect faces in real-time. 