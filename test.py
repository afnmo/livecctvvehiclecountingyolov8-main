

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import*
model=YOLO('yolov8s.pt')
# 3.8
# pip3.8


stream = CamGear(source='https://www.youtube.com/watch?v=En_3pkxIJRM', stream_mode = True, logging=True).start() # YouTube Video URL as input

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
tracker =Tracker()
area1 = [(550, 287), (441, 338), (464,352), (571, 295)]
area2 = [(584, 301), (480, 361), (500, 374), (610, 310)]
downcar = {} #dictionry
downcarcounter = []
upcar = {}
upcarcounter = []

while True:    
    frame = stream.read()   
    count += 1
    if count % 2 != 0:
        continue


    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id1=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        result = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        if result>=0: # if there is a car in the area
            downcar[id1] = (cx,cy)
        if id1 in downcar:
            result1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if result1 >=0:
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2) 
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                # count car once
                if downcarcounter.count(id1) == 0:
                    downcarcounter.append(id1)
##############################################################################              
    result2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
    if result2>=0: # if there is a car in the area
        upcar[id1] = (cx,cy)
    if id1 in upcar:
        result3 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        if result3 >=0:
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2) 
            cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
            # count car once
            if upcarcounter.count(id1) == 0:
                upcarcounter.append(id1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0,255,0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255,0,0), 2)
    print(downcar)
    cardown = len(downcarcounter)
    carup = len(upcarcounter)
    cvzone.putTextRect(frame,f'DownsideLane: {cardown}',(50,60),1,1)
    cvzone.putTextRect(frame,f'UpsideLane: {carup}',(846,59),1,1)
    # print(len(downcarcounter))
    # print(len(upcarcounter))
    cv2.imshow("RGB", frame)
    

    if cv2.waitKey(1)&0xFF==27:
        break
# cap.release()
cv2.destroyAllWindows()


