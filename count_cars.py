from flask import Flask, jsonify
from flask_socketio import SocketIO
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear
from tracker import Tracker
import time  # Import the time module

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize YOLO model and video stream
model = YOLO('yolov8s.pt')
stream = CamGear(source='https://www.youtube.com/watch?v=En_3pkxIJRM', stream_mode=True, logging=True).start()

# stream1 = enter
# stream2 = exit

# Load class list from coco.txt
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Initialize other variables
tracker = Tracker()
area1 = [(550, 287), (441, 338), (464, 352), (571, 295)]
area2 = [(584, 301), (480, 361), (500, 374), (610, 310)]
downcar = {}
downcarcounter = []
upcar = {}
upcarcounter = []

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('get_car_counts')
def get_car_counts():
    # Initialize variables to store car counts
    total_downside_count = 0
    total_upside_count = 0
    
    # Process video stream and count cars for 1 minute
    start_time = time.time()  # Start time of the counting process
    while time.time() - start_time <= 60:  # Count cars for 1 minute
        count_cars()
        # Update the total counts
        total_downside_count = len(downcarcounter)
        total_upside_count = len(upcarcounter)

    # Emit car counts to clients after 1 minute
    socketio.emit('car_counts', {'downside_lane': total_downside_count, 'upside_lane': total_upside_count})


def count_cars():
    global downcar, downcarcounter, upcar, upcarcounter

    # Read frame from video stream
    frame = stream.read()

    # Process frame for car detection
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Detect cars and update counters
    list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        result = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        if result >= 0:
            downcar[id1] = (cx, cy)
        if id1 in downcar:
            result1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if result1 >= 0:
                if downcarcounter.count(id1) == 0:
                    
                    downcarcounter.append(id1)
    result2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
    if result2 >= 0:
        upcar[id1] = (cx, cy)
    if id1 in upcar:
        result3 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        if result3 >= 0:
            if upcarcounter.count(id1) == 0:
                upcarcounter.append(id1)

if __name__ == '__main__':
    socketio.run(app, debug=True)
