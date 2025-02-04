import numpy as np
import matplotlib.pyplot as plt
a = np.array([1, 2, 3, 4])
#Hello world
#One reason numpy is superior to matlab is that it is able to operate on matrices or vectors in place of 
# of one element at a time. Another reason is numpy works on blocks of data at a time 
#instead of one piece of data. It may also utilize parralelization if possible
#The name of the data type is ndarray. It restricts the types of data that can be entered into vectors. 
#It also allocates compactly memory, where as python may not. This means when data is accessed, less processing time is required.
a1 = np.array([1, 2, 3, 4])
print(a1)
a2 = np.ones((3, 4))
print(a2)
a3 = np.zeros((4, 3))
print(a3)
a51 = np.ones((2,3), dtype=int)
a51 *= 3
a52 = np.ones((3,4), dtype=int)
a52 *= 2
a5 = a51 @ a52
print(a5)
a6 = np.array([[3, 1],[1, 2]])
print(a6)
eigenvalues, eigenvectors = np.linalg.eig(a6)
print(eigenvalues)
print(eigenvectors)

x = np.linspace(0, 2 * np.pi, 500)
y = np.sin(x)

plt.plot(x, y, label="sin(x)")
plt.xlabel("x (radianes)")
plt.ylabel("sin(x)")
plt.title("Graph of sin(x) en [0, 2π]")
plt.grid(True)
plt.legend()
plt.figure(1)
plt.show()


from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()

a31 = ([[3,1],[1,2]])
b31 = ([[9],[8]])
sol3 = np.linalg.solve(a31, b31)
print(sol3)

from scipy.fft import fft, fftfreq
N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(100.0 * 2.0*np.pi*x) + 0.5*np.sin(160.0 * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.figure(3)
plt.show()

from scipy.optimize import minimize

def func(x):
    return x**2 + 2*x
result = minimize(func, x0=0)
a34 = result.fun
print(a34)

import cv2

ip = "/Users/charlessullivan/ENEE408ISpring2025/2025_Spring_Lab_1/ENEE408ILab1Pic.jpg" 
i = cv2.imread(ip)

gi = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

plt.figure(4)
plt.imshow(gi, cmap='gray')
plt.axis("off")
plt.show()

edges = cv2.Canny(gi, 100, 200)

plt.figure(5)
plt.imshow(edges, cmap='gray')
plt.axis("off")
plt.show()

face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face.detectMultiScale(gi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

outlines = i.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(outlines, (x, y), (x + w, y + h), (255, 0, 0), 2)

plt.figure(6)
plt.imshow(cv2.cvtColor(outlines, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
