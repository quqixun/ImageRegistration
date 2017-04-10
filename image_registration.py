# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/10
#


import numpy as np
from affine_transform import Affine
from affine_ransac import Ransac
from PIL import Image
from matplotlib import pyplot as plt


# Affine Transform
# |x'|  = |a, b| * |x|  +  |tx|
# |y'|    |c, d|   |y|     |ty|
# pts_t =    A   * pts_s  + t


# Create instance
af = Affine()

# Generate a test case as validation with
# a rate of outliers
outlier_rate = 0.9
A_true, t_true, pts_s, pts_t = af.create_test_case(outlier_rate)

# At least 3 corresponding points to
# estimate affine transformation
K = 3
# Randomly select 3 pairs of points to do estimation
idx = np.random.randint(0, pts_s.shape[1], (K, 1))
A_test, t_test = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

# Display known parameters with estimations
# Ther are should be same when outlier_rate equals to 0
# print(A_true, '\n', t_true)
# print(A_test, '\n', t_test)

# Create instance
rs = Ransac(K=3, threshold=1)

# residual = rs.residual_lengths(A_test, t_test, pts_s, pts_t)

# Run RANSAC to estimate affine tansformation when
# many outliers in points set
# A_rsc, t_rsc, inliers = rs.ransac_fit(pts_s, pts_t)
# print(A_rsc, '\n', t_rsc)

# Load source image and target image
img_source = Image.open('Images/mona_source.png')
img_source = np.array(img_source)[:, :, :3]
img_target = Image.open('Images/mona_target.jpg')
img_target = np.array(img_target)[:, :, :3]

print(img_source.shape)
print(img_target.shape)

plt.subplot(121)
plt.imshow(img_source)
plt.axis('off')

plt.subplot(122)
plt.imshow(img_target)
plt.axis('off')
plt.show()
