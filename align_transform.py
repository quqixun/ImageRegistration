# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/11
#


import cv2
import numpy as np
from affine_ransac import Ransac
from affine_transform import Affine


# The ration of the best match over second best match
#      distance of best match
# ------------------------------- <= MATCH_RATIO
#  distance of second best match
RATIO = 0.8


class Align():

    def __init__(self, source_path, target_path,
                 K=3, threshold=1):
        ''' __INIT__

            Initialize the instance.

            Input arguments:

            - source_path : the path of sorce image that to be warped
            - target_path : the path of target image
            - K : the number of corresponding points, default is 3
            - threshold : a threshold determins which points are outliers
            in the RANSAC process, if the residual is larger than threshold,
            it can be regarded as outliers, default value is 1

        '''

        self.source_path = source_path
        self.target_path = target_path
        self.K = K
        self.threshold = threshold

    def read_image(self, path, mode=1):
        ''' READ_IMAGE

            Load image from file path.

            Input arguments:

            - path : the image to be read
            - mode : 1 for reading color image, 0 for grayscale image
            default is 1

            Output:

            - the image to be processed

        '''

        return cv2.imread(path, mode)

    def extract_SIFT(self, img):
        ''' EXTRACT_SIFT

            Extract SIFT descriptors from the given image.

            Input argument:

            - img : the image to be processed

            Output:

            -kp : positions of key points where descriptors are extracted
            - desc : all SIFT descriptors of the image, its dimension
            will be n by 128 where n is the number of key points


        '''

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract key points and SIFT descriptors
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(img_gray, None)

        # Extract positions of key points
        kp = np.array([p.pt for p in kp]).T

        return kp, desc

    def match_SIFT(self, desc_s, desc_t):
        ''' MATCH_SIFT

            Match SIFT descriptors of source image and target image.
            Obtain the index of conrresponding points to do estimation
            of affine transformation.

            Input arguments:

            - desc_s : descriptors of source image
            - desc_t : descriptors of target image

            Output:

            - fit_pos : index of corresponding points

        '''

        # Match descriptor and obtain two best matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_s, desc_t, k=2)

        # Initialize output variable
        fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

        matches_num = len(matches)
        for i in range(matches_num):
            # Obtain the good match if the ration id smaller than 0.8
            if matches[i][0].distance <= RATIO * matches[i][1].distance:
                temp = np.array([matches[i][0].queryIdx,
                                 matches[i][0].trainIdx])
                # Put points index of good match
                fit_pos = np.vstack((fit_pos, temp))

        return fit_pos

    def affine_matrix(self, kp_s, kp_t, fit_pos):
        ''' AFFINE_MATRIX

            Compute affine transformation matrix by corresponding points.

            Input arguments:

            - kp_s : key points from source image
            - kp_t : key points from target image
            - fit_pos : index of corresponding points

            Output:

            - M : the affine transformation matrix whose dimension
            is 2 by 3

        '''

        # Extract corresponding points from all key points
        kp_s = kp_s[:, fit_pos[:, 0]]
        kp_t = kp_t[:, fit_pos[:, 1]]

        # Apply RANSAC to find most inliers
        _, _, inliers = Ransac(self.K, self.threshold).ransac_fit(kp_s, kp_t)

        # Extract all inliers from all key points
        kp_s = kp_s[:, inliers[0]]
        kp_t = kp_t[:, inliers[0]]

        # Use all inliers to estimate transform matrix
        A, t = Affine().estimate_affine(kp_s, kp_t)
        M = np.hstack((A, t))

        return M

    def warp_image(self, source, target, M):
        ''' WARP_IMAGE

            Warp the source image into target with the affine
            transformation matrix.

            Input arguments:

            - source : the source image to be warped
            - target : the target image
            - M : the affine transformation matrix

        '''

        # Obtain the size of target image
        rows, cols, _ = target.shape

        # Warp the source image
        warp = cv2.warpAffine(source, M, (cols, rows))

        # Merge warped image with target image to display
        merge = np.uint8(target * 0.5 + warp * 0.5)

        # Show the result
        cv2.imshow('img', merge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return

    def align_image(self):
        ''' ALIGN_IMAGE

            Warp the source image into target image.
            Two images' path are provided when the
            instance Align() is created.

        '''

        # Load source image and target image
        img_source = self.read_image(self.source_path)
        img_target = self.read_image(self.target_path)

        # Extract key points and SIFT descriptors from
        # source image and target image respectively
        kp_s, desc_s = self.extract_SIFT(img_source)
        kp_t, desc_t = self.extract_SIFT(img_target)

        # Obtain the index of correcponding points
        fit_pos = self.match_SIFT(desc_s, desc_t)

        # Compute the affine transformation matrix
        M = self.affine_matrix(kp_s, kp_t, fit_pos)

        # Warp the source image and display result
        self.warp_image(img_source, img_target, M)

        return
