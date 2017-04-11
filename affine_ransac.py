# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/10
#


import numpy as np
from affine_transform import Affine


class Ransac():

    def __init__(self, K=3, threshold=1):
        self.K = K
        self.threshold = threshold

    def residual_lengths(self, A, t, pts_s, pts_t):
        if not(A is None) and not(t is None):
            pts_e = np.dot(A, pts_s) + t

            diff_square = np.power(pts_e - pts_t, 2)
            residual = np.sqrt(np.sum(diff_square, axis=0))
        else:
            residual = None

        return residual

    def ransac_fit(self, pts_s, pts_t):
        af = Affine()

        inliers_num = 0

        A = None
        t = None
        inliers = None

        for i in range(2000):
            idx = np.random.randint(0, pts_s.shape[1], (self.K, 1))
            A_tmp, t_tmp = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])
            residual = self.residual_lengths(A_tmp, t_tmp, pts_s, pts_t)

            if not(residual is None):
                inliers_tmp = np.where(residual < self.threshold)
                inliers_num_tmp = len(inliers_tmp[0])

                if inliers_num_tmp > inliers_num:
                    inliers_num = inliers_num_tmp
                    inliers = inliers_tmp
                    A = A_tmp
                    t = t_tmp
            else:
                pass

        return A, t, inliers
