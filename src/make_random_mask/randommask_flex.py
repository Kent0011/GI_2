import cv2
import numpy as np
import random
from tqdm import tqdm
import os

#================================================================

SIZE = 1000 # 画像サイズ
BLOCK = 50 # パターンブロックサイズ
NUM_OF_IMAGE = 1
VARIABLE = True # True: 可変, False: 固定
CUT_SIZE = 20
FILETYPE = '.bmp'
FILENAME = f'/Users/kent/Desktop/GI_2/mask/randomimage_size{SIZE}_block{BLOCK}_num{NUM_OF_IMAGE}/'

#================================================================


def makeRandomMask(size, filename, num, block):
    
    rand_size = size
    
    if VARIABLE == True: rand_size += CUT_SIZE
        
    width, height = rand_size, rand_size
    
    # filenameが存在しないとき作成
    if not os.path.exists(filename):
        os.makedirs(filename)
    
    # ランダムパターンを作成
    Random_img = np.zeros((height,width), dtype=np.uint8)
    for i in range(0, height, block):
        for j in range(0, width, block):
            if random.randint(1,100) <= 50:
                Random_img[i:i+block, j:j+block] = 0
            else:
                Random_img[i:i+block, j:j+block] = 255
    
    if VARIABLE == True:
        # パターンを規定サイズにカット
        r = random.randint(0, CUT_SIZE)
        img = Random_img[r:r+size, r:r+size]

    maskName = filename + str(num) + FILETYPE
    cv2.imwrite(maskName, img)



if __name__ == '__main__':
    size = SIZE
    block = BLOCK
    filename = FILENAME
    
    for i in tqdm(range(NUM_OF_IMAGE)):
        makeRandomMask(size, filename, i, block)
