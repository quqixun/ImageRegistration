# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/10
#


import numpy as np
from affine_transform import Affine


# The number of iterations in RANSAC
ITER_NUM = 2000


class Ransac():

    def __init__(self, K=3, threshold=1):
        ''' __INIT__

            Initialize the instance.

            Input argements:

            - K : the number of corresponding points,
            default is 3
            - threshold : determing which points are inliers
            by comparing residual with it

        '''

        self.K = K
        self.threshold = threshold

    def residual_lengths(self, A, t, pts_s, pts_t):
        ''' RESIDUAL_LENGTHS

            Compute residual length (Euclidean distance) between
            estimation and real target points. Estimation are
            calculated by the given source point and affine
            transformation (A & t).

            Input arguments:

            - A, t : the estimated affine transformation calculated
            by least squares method
            - pts_s : key points from source image
            - pts_t : key points from target image

            Output:

            - residual : Euclidean distance between estimated points
            and real target points

        '''

        if not(A is None) and not(t is None):
            # Calculate estimated points:
            # pts_esti = A * pts_s + t
            pts_e = np.dot(A, pts_s) + t

            # Calculate the residual length between estimated points
            # and target points
            diff_square = np.power(pts_e - pts_t, 2)
            residual = np.sqrt(np.sum(diff_square, axis=0))
        else:
            residual = None

        return residual

    def ransac_fit(self, pts_s, pts_t):
        ''' RANSAC_FIT

            Apply the method of RANSAC to obtain the estimation of
            affine transformation and inliers as well.

            Input arguments:

            - pts_s : key points from source image
            - pts_t : key points from target image

            Output:

            - A, t : estimated affine transformation
            - inliers : indices of inliers that will be applied to refine the
            affine transformation

        '''

        # Create a Affine instance to do estimation
        af = Affine()

        # Initialize the number of inliers
        inliers_num = 0

        # Initialize the affine transformation A and t,
        # and a vector that stores indices of inliers
        A = None
        t = None
        inliers = None

        for i in range(ITER_NUM):
            # Randomly generate indices of points correspondences
            idx = np.random.randint(0, pts_s.shape[1], (self.K, 1))
            # Estimate affine transformation by these points
            A_tmp, t_tmp = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

            # Calculate the residual by applying estimated transformation
            residual = self.residual_lengths(A_tmp, t_tmp, pts_s, pts_t)

            if not(residual is None):
                # Obtain the indices of inliers
                inliers_tmp = np.where(residual < self.threshold)
                # Obtain the number of inliers
                inliers_num_tmp = len(inliers_tmp[0])

                # Set affine transformation and indices og inliers
                # in one iteration which has the most of inliers
                if inliers_num_tmp > inliers_num:
                    # Update the number of inliers
                    inliers_num = inliers_num_tmp
                    # Set returned value
                    inliers = inliers_tmp
                    A = A_tmp
                    t = t_tmp
            else:
                pass

        return A, t, inliers
