## jrana 2025-01-28
## Lab Tutorial 2
## 3.2 NumPy
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, optimize, fft
import cv2 as cv

# 1. Why is numpy faster than using for loops in Python for operations such as matrix multiplications?
# Ans: NumPy is faster than using for loops in Python because it implements matrix multiplications and other complex operations in C. This bypasses the overhead costs of using for-loops in Python and also enables more optimized algorithms to be used. As a result, NumPy can perform operations significantly faster than regular Python code.

# 2. What is the data type used in numpy which makes such operations feasible? Also name a few differences between this data type and its similar counterpart in Python.
# Ans: NumPy uses NumPy arrays in order to make these operations feasible. Its Python counterpart is the list. NumPy arrays can only hold one type of data, while lists can hold different types together. NumPy arrays store their data in a contiguous block of memory, which reduces overhead for storing locations. Meanwhile, list elements can be non-contiguous, which requires more overhead to determine the location of each element. Finally, since NumPy arrays can only hold one datatype, there is less data that you need to store for each element. In lists, you need to store the datatype of each element.

# 3. Go through the documentation and create a numpy array with the elements [1,2,3,4].
print("\n~~~~~PROBLEM 3~~~~~")
a = np.array([1, 2, 3, 4])
print(a)

# 4. Use np.ones, np.zeros to create an array of 1’s with dimension 3x4 and an array of 0’s with dimension 4x3.
print("\n~~~~~PROBLEM 4~~~~~")
a = np.ones((3, 4))
print(a)
b = np.zeros((4, 3))
print(b)

# 5. Create a 2x3 matrix A and a 3x4 matrix B and perform a matrix multiplication using numpy.
print("\n~~~~~PROBLEM 5~~~~~")
a = np.array([(1, 2, 3), (4, 5, 6)])
print("A = ", a)
b = np.array([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
print("B = ", b)
c = a @ b
print("A @ B (matrix multiplication) = ", c)

# 6. Find the eigenvalues and eigenvectors of the matrix given below?
print("\n~~~~~PROBLEM 6~~~~~")
a = np.array([(3, 1), (1, 2)])
print("A = ", a)
a_eigenvalues, a_eigenvectors = np.linalg.eig(a)
print("Eigenvalues of A: ", a_eigenvalues)
print("Eigenvectors of A: ", a_eigenvectors)


## 3.3 Matplotlib

# 1. Create a line plot of the sine function over the interval [0, 2π] using Matplotlib.
print("\n~~~~~PROBLEM 1~~~~~")
fig, ax = plt.subplots()
x = np.arange(0.0, 2*np.pi, 0.01)
y = np.sin(x)
ax.plot(x, y)
plt.show()

# 2. Add labels to the axes in a Matplotlib plot.
print("\n~~~~~PROBLEM 2~~~~~")
fig, ax = plt.subplots()
x = np.arange(0.0, 2*np.pi, 0.01)
y = np.sin(x)
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")
ax.set_title("Graph of sin(x) over [0, 2pi]")
plt.show()

# 3. Plot the 3d graph of the function given below using Matplotlib.
print("\n~~~~~PROBLEM 3~~~~~")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
xs, ys = np.meshgrid(x, y)
zs = np.sin(np.sqrt(np.square(xs) + np.square(ys)))
ax.plot_surface(xs, ys, zs)
plt.show()


## 3.4 SciPy

# 1. Solve the linear system of equations given below using SciPy.
print("\n~~~~~PROBLEM 1~~~~~")
a = np.array([(3, 1), (1, 2)])
b = np.array([9, 8])
x = linalg.solve(a, b)
print("Solution: (x, y) = ", x)
print("Verify: plugging in solution into equations, result is: ", np.dot(a, x))

# 2. Find the minimum of the function given below using SciPy's optimization module.
print("\n~~~~~PROBLEM 2~~~~~")
def f(x):
    return x*x + 2*x # y = x^2 + 2x
min = optimize.minimize_scalar(f)
print("The minimum value of y = x^2 + 2x is y =", min.fun)
print("y reaches this minimum value at x = ", min.x)

# 3. Perform the Fourier transformation of the function given below using SciPy. Plot the frequency response using matplotlib
# Following this tutorial: https://docs.scipy.org/doc/scipy/tutorial/fft.html
print("\n~~~~~PROBLEM 3~~~~~")
N = 2000 # number of samples
T = 2.0*np.pi/2000 # sampling frequency
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(100*np.pi*x) + 0.5*np.sin(160*np.pi*x)
yf = fft.fft(y)
xf = fft.fftfreq(N, T)
plt.plot(xf, 2.0/N * np.abs(yf))
plt.show()


## 3.5 OpenCV

# 1. Read an image and convert it to grayscale using OpenCV.
print("\n~~~~~PROBLEM 1~~~~~")
original_img_filename = "jrana_lab_1_rocky.jpg"
original_img = cv.imread(cv.samples.findFile(original_img_filename), cv.IMREAD_COLOR)
cv.imshow("Original Image", original_img)
gray_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale Image", gray_img)
cv.waitKey(0)
cv.destroyAllWindows()

# 2. Perform edge detection on the image using OpenCV.
print("\n~~~~~PROBLEM 2~~~~~")
cv.imshow("Original Image", original_img)
edge_img = cv.Canny(gray_img, 60, 115)
cv.imshow("Image with Edge Detection", edge_img)
cv.waitKey(0)
cv.destroyAllWindows()

# 3. Use a Haar cascade classifier to implement face detection using OpenCV on an image which contains faces.
# Following this tutorial: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
print("\n~~~~~PROBLEM 2~~~~~")
face_img_filename = "jrana_lab_1_face.jpg"
face_img = cv.imread(cv.samples.findFile(face_img_filename), cv.IMREAD_COLOR)
cv.imshow("Original Image", face_img)
face_gray_img = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "/haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(face_gray_img, 1.1, 9)
for (x, y, w, h) in faces:
    center = (x + w//2, y + h//2)
    face_img = cv.ellipse(face_img, center, (w//2, h//2), 0, 0, 360, (255, 0, 0), 4)
cv.imshow("Image with Face Detection", face_img)
cv.waitKey(0)
cv.destroyAllWindows()

# Homework: Explain how Haar cascades work to detect faces.
# Ans: A Haar Cascade is a series of "weak" classifications chained (cascaded) together in order to provide a "strong" classification of an image. The weak classifications are generated via machine learning, and then each classification is applied to the training data to determine which features provide the least error. Then, the classifiers are grouped and chained together such that an image only moves to the next group if it matches a feature in the previous group. This drastically reduces the number of classifiers and amount of computation that is required to process an image.
# Source: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html