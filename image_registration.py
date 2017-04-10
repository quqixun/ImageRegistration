# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/10
#


import numpy as np
from affine_transform import Affine


# Affine Transform
# |x'|  = |a, b| * |x|  +  |tx|
# |y'|    |c, d|   |y|     |ty|
# pts_t =    A   * pts_s  + t


# Create instance
af = Affine()

# Generate a test case as validation with
# a rate of outliers
outlier_rate = 0
A_true, t_true, pts_s, pts_t = af.create_test_case(0)

# At least 3 corresponding points to
# estimate affine transformation
K = 3
# Randomly select 3 pairs of points to do estimation
idx = np.random.randint(0, pts_s.shape[1], (K, 1))
A_test, t_test = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

# Display known parameters with estimations
# Ther are should be same when outlier_rate equals to 0
print(A_true, '\n', t_true)
print(A_test, '\n', t_test)
