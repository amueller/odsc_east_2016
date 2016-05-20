import numpy as np
from sklearn.linear_model import LinearRegression


X = np.array([[66, 1882,   19, 10030,  2],
              [52, 1337,   2 , 10028,  0],
              [22, 3467,   8 , 10041,  1],
              [25, 8391,   27, 10009,  4]], dtype=np.int)

#X = np.array([[65, 2, 19, 30, 22],
#              [51, 22, 2, 28, 26],
#              [21, 32, 8, 41, 9],
#              [24, 1, 47, 19, 1]])

y = [50500, 41000, 35200, 36000]

lr = LinearRegression().fit(X[:-1], y[:-1])

print("predictions: %s" % (np.dot(X, (lr.coef_).astype(np.int)) + (lr.intercept_).astype(np.int)))
print("coefficients: %s" % (lr.coef_).astype(np.int))
print("intercept: %s" % (lr.intercept_).astype(np.int))

print("\nusing only age")

lr = LinearRegression().fit(X[:-1, :1], y[:-1])

print("predictions: %s" % (np.dot(X[:, :1], (lr.coef_).astype(np.int)) + (lr.intercept_).astype(np.int)))
print("coefficients: %s" % (lr.coef_).astype(np.int))
print("intercept: %s" % (lr.intercept_).astype(np.int))
