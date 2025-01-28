# importing the modules
import numpy as np
import timeit

# vectorized sum
print(np.sum(np.arange(15000)))

print("Time taken by vectorized sum : ", end = "")
timeit.timeit(stmt='np.sum(np.arange(15000));',setup="import numpy as np")

# iterative sum
total = 0
for item in range(0, 15000):
    total += item
a = total
print("\n" + str(a))

print("Time taken by iterative sum : ", end = "")
timeit.timeit(stmt=str(a), number=100000)