from fileinput import filename
import cv2
import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

# =====================================================

NUMBER = 5000
OBJECT_PASS = "/Users/kent/Desktop/GI_2/image/5*5dots_random.bmp"
MASK_PASS = "/Users/kent/Desktop/GI_2/mask/randomimage_size200_block1_num5000/"
SIZE = 200
RESULT_PASS = "/Users/kent/Desktop/GI_2/result_image"


# =====================================================

def calcCGI(num,objname, maskname, size, resultfolder):
    B = 0
    I = np.zeros((size, size))
    BI = np.zeros((size, size))
    T2 = np.zeros((size, size))

    img = cv2.imread(objname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    
    for i in range(num):  
        mask = cv2.cvtColor(cv2.imread(maskname + str(i) + '.bmp'), cv2.COLOR_BGR2GRAY)
        mask = mask / 255

        In = np.multiply(img,mask) #要素同士をかけ合わせてる

        light_value = 0
        # 画素値から強度を計算する
        light_value = np.sum(In)

        #Bはスカラー、 
        B = B + light_value / num

        # IとBIの計算
        I = I + mask / num
        BI = BI + mask * light_value / num
        
    # 再構成画像出力
    tmp = BI - B*I
    max = np.max(tmp)
    min = np.min(tmp)

    T2 = (tmp - min) / (max - min)

    #0-1から255階調に変更
    T2 = np.clip(T2 * 255, 0, 255).astype(np.uint8)

    if not os.path.exists(resultfolder):
        os.makedirs(resultfolder)
    resultfile = resultfolder + "GI_simuration_" +"random9.bmp"
    
    if cv2.imwrite(resultfile, T2):
        print('simuration succed')



calcCGI(NUMBER, OBJECT_PASS, MASK_PASS, SIZE, RESULT_PASS)
