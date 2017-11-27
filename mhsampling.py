import numpy as np
from scipy.stats import multivariate_normal

id3 = np.identity(3)
sigma_l = 6*id3
mean_l = np.array([0,0,4]).T
id2 = np.identity(2)
sigma_2d = 0.05*0.05*id2
ratios = np.genfromtxt('data/inputs.csv',delimiter=',')
points = np.genfromtxt('data/points_2d_camera_1.csv',delimiter=',')

vars = multivariate_normal.pdf()
print(vars)