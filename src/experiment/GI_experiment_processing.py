import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from tqdm import tqdm


#GIの計算
#積分球により計測したBの値を用いて再構成
def calcCGI(measurment_num, size, b_path, mask_folder, save_folder):
    start = time.time()
    I = np.zeros((size, size))
    BI = np.zeros((size, size))

    T = np.zeros((size, size))
    T1 = np.zeros((size, size))
    T2 = np.zeros((size, size))

    f = open(b_path, 'r')
    B = 0
    height = size
    width = size

    # Bの正規化
    bs = []
    min = 0
    
    '''
    for i in range(measurment_num):
        Bvalue = f.readline()
        light_value = float(Bvalue)
        bs.append(light_value)
    min = np.min(bs)
    print('min:', min)
    f.close()
    '''

    f = open(b_path, 'r')
    for i in tqdm(range(measurment_num)):

        mask = cv2.imread(mask_folder + str(i) + '.bmp', cv2.IMREAD_GRAYSCALE)
        mask = 255 - mask
        mask = mask / 255

        Bvalue = f.readline()
        light_value = float(Bvalue)
        B = B + light_value / measurment_num
        square_x = (width - size) // 2
        square_y = (height - size) // 2

        mask_sl = mask[square_y:square_y+size, square_x:square_x+size]
   
        I = I + mask_sl / measurment_num
        BI = BI + (mask_sl*light_value / measurment_num)
        
                
    tmp = BI - B*I
    min = np.min(tmp)
    max = np.max(tmp)

    T2 = (BI - B*I - min)/(max - min)

    T2a = np.clip(T2 * 255, 0, 255).astype(np.uint8)

    cv2.imwrite(save_folder + 'block5_' + str(measurment_num) +'.png', T2a)

    f.close()



# 変更箇所
calcCGI(
    measurment_num=200,
    size = 200,
    b_path = "/Users/kent/Desktop/GI_2/result_text/block5_0117_5000.txt",
    mask_folder='/Users/kent/Desktop/Kenkyu/GI/2024_9_10/randomimage_size200_block5_num5000/',
    save_folder='/Users/kent/Desktop/GI_2/result_text/')




