from ultralytics import YOLO
#using the model v8 and then will fine tune it for airbnb specific dataset 
model = YOLO("yolov8n.pt")

# training on custom dataset with batch size =4

model.train(data="../data/data.yaml", epochs=10 ,batch=4, imgsz=640)

#saving the trained fine tuned  model

model.save("../models/best.pt")