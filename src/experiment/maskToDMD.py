import cv2
from tqdm import tqdm
import numpy as np

dir = "/Users/kent/Desktop/GI_2/mask/randomimage_size200_block5_num5000/"
WHITE = [255, 255, 255]
input_size = 200
output_width = 1920
output_height = 1080

add_hight = (output_height-input_size)//2 
add_width = (output_width-input_size)//2

for i in tqdm(range(5000)):
    img_path = dir + str(i) + '.bmp'
    img = cv2.imread(img_path)

    # 520x520の0配列（黒）
    # img = [[255 for j in range(520)] for i in range(520)]
    # img = np.array(img)
    # img[50:150, 50:150] = 0

    # print(img)
    new_img = cv2.copyMakeBorder(img, add_hight-12, add_hight+12, add_width-12, add_width+12, cv2.BORDER_CONSTANT, value=WHITE)
    text = str(i)
    cv2.putText(new_img, text=text, org=(40, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX,  fontScale=2.0,color=(0,0,0),thickness=5, lineType=cv2.LINE_4)
    # cv2.putText(new_img, text=text, org=(900, 700), fontFace=cv2.FONT_HERSHEY_COMPLEX,  fontScale=15.0,color=(0,0,0),thickness=20, lineType=cv2.LINE_4)

    save_path = "/Users/kent/Desktop/GI_2/mask/DMD/"+ str(i) + '.bmp'
    # save_path = 'D:/GI2024/DMD check images/520-1.png'
    cv2.imwrite(save_path, new_img)
