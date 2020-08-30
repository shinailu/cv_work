import numpy as np
import cv2
import sys
import base64

#没有输入文件夹地址，默认文件夹
#filepath = sys.argv[1] if sys.argv[1:] else 'rtsp://admin:a26608679@10.0.111.80:554/h264/ch1/main/av_stream'
cap = cv2.VideoCapture('rtsp://admin:a26608679@10.0.111.80:554/h264/ch1/main/av_stream')
#cap.set(cv2.CAP_PROP_POS_FRAMES, float(1))
print('is opened',cap.isOpened())
if cap.isOpened(): #判断是否正常打开
    rval, frame = cap.read()

    img_str = cv2.imencode(".jpg", frame)[1].tostring()
    image_code = str(base64.b64encode(img_str))[2: -1]
    print(image_code)
    cap.release()