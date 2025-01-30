# Secition 3.2 NumPy

# Question 1: numpy is faster than using for loops in Python for operations such as matrix multiplications because numpy was designed to process large data sets such as matricies or vectors. 
#   Compared to python which needs to compile the source code and then feed the output to an interpreter to be executed. This is very inefficient compared to numpy because numpy knows what you are trying to do and has optimized for it.

# Question 2: The main datatype used in numpy is a ndarray. This array has a few more attributes/values stored in a tuple which allows numpy to add more funcationallity compared to python arrays. 

# Question 3:
print("Section 3.2 Numpy:\n")
import numpy as np
import scipy.fft
import scipy.fftpack
import scipy.linalg
import scipy.optimize
a = np.array([1, 2, 3, 4])
print("Question 3: " + str(a))

# Question 4:
ones_array = np.ones((3, 4))
zeros_array = np.zeros((4, 3))
print("\nQuestion 4:\nOnes Array:\n" + str(ones_array) + "\nZeros Array:\n"+ str(zeros_array))

# Question 5
A = np.array([[1, 1, 1], [2, 2, 2]])
B = np.array ([[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]])
result = np.matmul(A, B)
print("\nQuestion 5:\nMatrix A:\n" + str(A) + "\nMatrix B\n" + str(B) + "\nProduct of Matrix A and B:\n" + str(result))


# Question 6
q6_matrix = np.array([[3, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(q6_matrix)
print("\nQuestion 6:\nEigen values: " + str(eigenvalues) + "\nEigenvectors: " + str(eigenvectors))


# Secition 3.3 Matplotlib
import matplotlib.pyplot as plt

# Question 1
x = np.linspace(0, 2*np.pi, 200)
plt.plot(x, np.sin(x), label='sin')
plt.title("Section 3.3, Question 1 and 2")

# Question 2
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()

# Question 3
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.linspace(-2*np.pi, 2*np.pi, 1000)
x, y = np.meshgrid(x, y)
ax = plt.figure().add_subplot(projection='3d')

ax.plot_surface(x, y, np.sin(np.sqrt(x**2 + y**2)))
ax.set_title("Section 3.3, Question 3")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()


# Section 3.4 SciPy

# Question 1
print("\nSection 3.4 SciPy\n")
import scipy
a = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
solution = scipy.linalg.solve(a, b)

print("Question 1: " + str(solution))

# Question 2
fun = lambda x: (x**2 + 2*x)
result = scipy.optimize.minimize(fun, 1)
print("Question 2: Minimum: " + str(result.x))

# Question 3
N = 600
T = 1/800
x = np.linspace(0, N*T, N)
y = np.sin(100*np.pi*x) + 1/2*np.sin(160*np.pi*x)
yf = scipy.fftpack.fft(y)
xf = np.linspace(0, 1/(2*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
ax.set_xlabel("Frequency")
ax.set_ylabel("Amplitude")
ax.set_title("Section 3.4, Question 3")
plt.show()

# Section 3.5 OpenCV

# Question 1
import cv2

image = cv2.imread("./image.png")
grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey_img = cv2.resize(grey_img, (960, 540))

# Question 2
edges = cv2.Canny(grey_img, 100, 200)
plt.figure(figsize=(12,6))
plt.subplot(121),plt.imshow(grey_img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# Question 3
# With help from https://www.geeksforgeeks.org/face-detection-using-cascade-classifier-using-opencv-python/
img = cv2.imread("./image2.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 2) 
for (x, y, w, h) in faces_rect: 
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 
cv2.imshow('Detected faces', img) 
cv2.waitKey(0) 

# The Haar Cascade works by first detecting differences in light and dark regions in an image. This can be used to detect edges or lines in an image.
# Then the Haar Cascade chains these classifiers in series which each disregard or consider certian features in the image.
# The resulting features left in the image should be the features you want or the classifier is trained on. 