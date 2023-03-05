# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/10
#
import time

import numpy as np
import cv2
from affine_ransac import Ransac
from align_transform import Align
from affine_transform import Affine


def img_regist(source_path, target_path):
    # Affine Transform
    # |x'|  = |a, b| * |x|  +  |tx|
    # |y'|    |c, d|   |y|     |ty|
    # pts_t =    A   * pts_s  + t

    # -------------------------------------------------------------
    # Test Class Affine
    # -------------------------------------------------------------

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
    # They should be same when outlier_rate equals to 0,
    # otherwise, they are totally different in some cases
    # print(A_true, '\n', t_true)
    # print(A_test, '\n', t_test)

    # -------------------------------------------------------------
    # Test Class Ransac
    # -------------------------------------------------------------

    # Create instance
    rs = Ransac(K=3, threshold=1)

    residual = rs.residual_lengths(A_test, t_test, pts_s, pts_t)

    # Run RANSAC to estimate affine tansformation when
    # too many outliers in points set
    A_rsc, t_rsc, inliers = rs.ransac_fit(pts_s, pts_t)
    # print(A_rsc, '\n', t_rsc)

    # -------------------------------------------------------------
    # Test Class Align
    # -------------------------------------------------------------

    # Load source image and target image

    al = Align(source_path, target_path, threshold=1)
    m_img = al.align_image()
    return m_img


if __name__ == '__main__':
    img_num = 10001
    path = 'D:/google downloads/experiment/image_registration_test/test/data/'
    # /content/drive/Othercomputers/惠普暗影精灵/华理(new)/组会实验/scalarFlow_dataset/raymarching/big_dataset/test/data/10001_img1.png
    # path = '/content/drive/Othercomputers/惠普暗影精灵/华理(new)/组会实验/scalarFlow_dataset/raymarching/big_dataset/test/data/'

    total_time_start = time.time()
    for i in range(500):

        # source_path = 'Images/mona_source.png'
        source_path = '{}_img2_all.png'.format(img_num + i)
        # target_path = 'Images/mona_target.jpg'
        target_path = '{}_img1.png'.format(img_num + i)

        time_start = time.time()

        m_img = img_regist(path+source_path, path+target_path)

        time_end = time.time()
        print('time cost', time_end - time_start, 's')

        # cv2.imwrite("img2_reg.png", m_img)
    total_time_end = time.time()
    print('total time cost', total_time_end - total_time_start, 's')
