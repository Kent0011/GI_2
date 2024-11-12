import os
import time
from datetime import datetime
import cv2
import requests
import json
import USBADSMS
import numpy as np

DevId = 0           #デバイスID 0~15
OpenState = True   #デバイスのオープン状態

# color define 入力値表示用
COL_RED = "#FFF000000"      #赤
COL_DRED = "#800000000"     #暗い赤

#/// Sub Functions //////////////////////////////

# Widgetの表示を設定する
# stat -> 'Close' , 'Open'

# 文字列strを表示する
def ShowInfo(str):
    print("",str)

# リターンコードretcodeに対応する文字列を表示する
def ShowRet(retcode):
    ShowInfo(USBADSMS.GetErrMessage(retcode))

# datの指定ビットbitが1(True)か0(False)を判定する
def IsBitSet(dat,bit):
    if( ( ( dat >> bit ) & 0b1 ) == 0b1 ): return True
    else: return False

# AD変換データから電圧値へ変換する
#オフセットテーブル
AdOffset = {
    "±10V":-10.0,
    "±5V":-5.0,
    "±2.5V":-2.5,
    "±1V":-1.0,
    "±0.5V":-0.5,
    "±0.25V":-0.25,
    "±0.1V":-0.1,
    "10V":0.0,
    "5V":0.0,
    "2.5V":0.0,
    "1V":0.0,
    "0.5V":0.0,
    "0.25V":0.0,
    "0.1V":0.0
}
#スパンテーブル
AdSpan = {
    "±10V":20.0,
    "±5V":10.0,
    "±2.5V":5.0,
    "±1V":2.0,
    "±0.5V":1.0,
    "±0.25V":0.5,
    "±0.1V":0.2,
    "10V":10.0,
    "5V":5.0,
    "2.5V":2.5,
    "1V":1.0,
    "0.5V":0.5,
    "0.25V":0.25,
    "0.1V":0.1
}

#/// Button event handlers /////////////////////////
#デバイスオープン
def Open_on():
    global DevId
    # DevId = id_combo.current()
    DevId = 0
    ret = USBADSMS.Tusbadsms_Device_Open(DevId)
    # if(ret == 0):
        # WidgetState('Open')
    # else:
        # ShowRet(ret)

#デバイスクローズ
# def Close_on():
    # USBADSMS.Tusbadsms_Device_Close(DevId)
    # WidgetState('Close')

def CnvtToVolt(addata):
    # range = range_combo.get()
    range = '0.1V'
    return addata * AdSpan[range] / 4096.0 + AdOffset[range]

'''
#デジタル出力
def DigOut_on():
    dob = 0b01 if(ValDo0.get()) else 0b00
    dob |= 0b10 if(ValDo1.get()) else 0b00
    ret = USBADSMS.Tusbadsms_Pio_Write(DevId,dob)
    if( ret):
        ShowRet(ret)

#デジタル入力
def DigIn_on():
    ret,dib = USBADSMS.Tusbadsms_Pio_Read(DevId)
    if( ret == 0 ):
        DiLed0['bg'] = COL_RED if(IsBitSet(dib,0)) else COL_DRED
        DiLed1['bg'] = COL_RED if(IsBitSet(dib,1)) else COL_DRED
    else:
        ShowRet(ret)
'''

#AD変換データ取得
def SingSmpl_on():
    # ch = ch_combo.current()
    ch = 0
    # range = range_combo.current()
    range = 13 
    ret,addat = USBADSMS.Tusbadsms_Single_Sample(DevId,ch,range)
    if( ret == 0 ):
        # VoltLbl["text"] = format(CnvtToVolt(addat),".4f")
        # print(f'addat:{format(addat)}')
        return (format(addat))
        
    else:
        ShowRet(ret)



#photo detector を用いて強度Biを取得するプログラム
def save_frame_camera_key(b_save_folder , measurment_num, mask_folder):

    b = []
    cv2.namedWindow('WinName', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('WinName', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    time_sta = time.time()
    for i in range(measurment_num):
    
        #DMDのパターンを選択
        imgName = mask_folder + str(i) + '.bmp'

        # 白黒のテストパターン
        """
        if(i%2 == 0):
            imgName = "D:/GI2024/DMD check images/black.png"
        else:
            imgName = "D:/GI2024/DMD check images/white.png"
        """
        
        rand_img = cv2.imread(imgName)
        
        cv2.imshow('WinName', rand_img)
        if i == 0:
            cv2.waitKey(0)
            # __ = SingSmpl_on()
        # print(i)
  
        # cv2.waitKey(300) #ISでも計測するなら
        cv2.waitKey(100) #Bのみ計測するときはこっち90
        time.sleep(0.01)#defoult 0.05

        # Bの計測
        values = []
        for j in range(31):
            values.append(float(SingSmpl_on()))
        b.append(np.median(values))
        # """
        # 配列の最大値と最小値を外れ値とみなして削除
        # del values[(np.argmax(values))]
        # del values[(np.argmin(values))]
        # """
        # print(values)
        # f.write(str(np.median(values)) + '\n')
        # time.sleep(0.01)#defoult 0.05
    
    cv2.destroyWindow('WinName')
    with open(b_save_folder  + 'block6_' + str(measurment_num) + '.txt', 'w') as f:
        for s in b:
            f.write("%s\n" % s)


    time_end = time.time()
    tim = int((time_end-time_sta) / 60)

    text="実験終了:"+str(measurment_num)+"回/n"+str(tim)+"分かかりました"
    # webhookURL
    webhook_url = "https://hooks.slack.com/services/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"

    # send to Slack
    requests.post(webhook_url, data = json.dumps({
        "text": text
    }))


Open_on()
# save_frame_camera_key(積分球強度保存フォルダ、イメージセンサー画像保存フォルダ、DMDマスクフォルダ、測定回数)
save_frame_camera_key(
    b_save_folder="//cs13S104/Public/2024th/yamamoto/GI/result/",
    measurment_num=5000,
    mask_folder="D:/GI2024yamamoto/randomimage_size200_block6_num5000/"
    )
