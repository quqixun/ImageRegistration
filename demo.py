import time

import numpy as np
import cv2

from image_registration import img_regist

if __name__ == '__main__':
    img_num = 10001
    path = 'D:/google downloads/experiment/image_registration_test/test/data/'
    # /content/drive/Othercomputers/惠普暗影精灵/华理(new)/组会实验/scalarFlow_dataset/raymarching/big_dataset/test/data/10001_img1.png
    # path = '/content/drive/Othercomputers/惠普暗影精灵/华理(new)/组会实验/scalarFlow_dataset/raymarching/big_dataset/test/data/'

    total_time_start = time.time()
    for i in range(5):

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
