import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import cv2

# Question 1: This is due to NumPy using vectorised operations, meaning it does not do operations to each individual element but rather applies 
# these to the entire matrix 

# Question 2: The data type that allows this is "ndarray." This is different to python as it is much faster than pythion because it is done in C
# As for a few specific differences, these are homogenious unike python lists as they must all be of the samedata type, also, Numpy 
# provides specific data types to allow for better memory allocation 

# Question 3 
array = np.array([1, 2, 3, 4])
print(arr)

# Question 4: 
ones = np.ones((3, 4))
print(ones)

zeros = np.zeros((4, 3))
print(zeros)


# Question 5: 
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

C = np.dot(A, B)


# Question 6: 
A = np.array([
    [3, 1],
    [1, 2]
])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)
print(eigenvectors)




## Section 2 MatPlotLib

# Part 1: 
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x, y)

# Part 2: 
plt.xlabel("x")
plt.ylabel("sin(x)")

plt.show()


# Part 3: 
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
X, Y = np.meshgrid(x, y)

Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

fig.colorbar(surface, shrink=0.5, aspect=8)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z = sin( sqrt(X^2 + Y^2) )")

plt.show()

## SciPy Part 3 

# Part 1: 
A = np.array([[3, 1],
              [1, 2]])

b = np.array([9, 8])

solution = solve(A, b)
x, y = solution

print(f"x,y")

# Part 2: 

def f(x):
    return x[0]**2 + 2*x[0]

initial_guess = np.array([0.0])

result = minimize(f, initial_guess)

print("Minimum value of x:", result.x[0])


# Part 3: 

Fs = 1000.0   
# Number of samples
N = 1000    

x = np.linspace(0, 1, N, endpoint=False)

f = np.sin(100*np.pi*x) + 0.5 * np.sin(160*np.pi*x)

F = fft(f)            
freqs = fftfreq(N, 1/Fs)

positive_freqs = freqs[:N//2]
magnitude_spectrum = np.abs(F)[:N//2]

plt.figure(figsize=(7, 4))
plt.plot(positive_freqs, magnitude_spectrum, 'b-')
plt.title("Frequency Response")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()



## Last Part 4
image = cv2.imread('mathew.weirich_cone.jpg')  

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

low_threshold = 100
high_threshold = 200
edges = cv2.Canny(gray, low_threshold, high_threshold)

# Part 3: 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    cv2.rectangle(
        image, 
        (x, y), 
        (x + w, y + h), 
        (0, 255, 0), 
        2
    )

cv2.namedWindow("Gray Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Edges", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Faces Detected", cv2.WINDOW_AUTOSIZE)

cv2.imshow("Gray Image", gray)
cv2.imshow("Edges", edges)
cv2.imshow("Faces Detected", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

### Homework: 

# A Haar cascade is a method that uses simple rectangular “Haar-like features” to detect faces by comparing intensity differences in specific regions of an image. 
# It uses an integral image for  feature calculation and arranges these features in a cascade of classifiers to quickly discard non-face regions and focus what it thinks may be faces. 
# This is an older approach to this but still works somewhat well, just not relative to modern methods 












