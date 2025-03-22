from ultralytics import YOLO
import cv2
from utils import speak_objects

#here loading our finetuned model that is best.pt 
model=YOLO("../models/best.pt")

#initialising the cv bascially the camera
# reading the frames from the camera
# passing the frames to the model for detection
# plotting the detections on the frames
# showing the frames with detections
# if q is pressed then breaking the loop

import time

        
        
def scan():

    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    last_spoken=[]
    frame_count=0
    while True:
        ret, frame=cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        if frame_count %15 == 0:
            start = time.time()
            results = model(frame)
            detected_objects = [model.names[int(box.cls[0])] for result in results if result.boxes for box in result.boxes]
            fps = 1 / (time.time() - start)
            last_spoken = speak_objects(detected_objects, last_spoken)
            frame = results[0].plot()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("object detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    scan()