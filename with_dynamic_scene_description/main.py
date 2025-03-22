#Tjis is mainly for a blind persons use case where the camera is used to detect objects and describe the scene to the user.
#The user can interact with the system by asking for a description of the scene or asking for help to navigate through obstacles.
#The system uses YOLO for object detection and Blip for scene description generation.
#The response is generated dynamically based on the detected objects and the scene description.
#The system also uses pyttsx3 for text-to-speech conversion to help the user interact with the system.
#The system continuously scans the environment and provides real-time feedback to the user.

from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import pyttsx3
from PIL import Image
import numpy as np
import random

#initialising yolo model (custom model if needed)
model=YOLO("yolov8n.pt") #replace with "best.pt" if using custom

#initialising blip for scene descriptions
processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#initialising text-to-speech engine
engine=pyttsx3.init()
engine.setProperty('rate',150)

#response generator class for dynamic and user friendly response 
class DynamicResponseGenerator:
    def __init__(self):
        self.memory=set() #store previously detected objects
        self.last_description="" #track last scene description
        self.templates={
            "greeting":[
                "Hello there! I see {}. Want me to describe more?",
                "Hi! I noticed {}. Should I help you interact with it?",
            ],
            "obstacle":[
                "Careful! A {} is nearby. Would you like assistance?",
                "There’s a {} in your path. Do you need help navigating?",
            ],
            "discovery":[
                "I see {} for the first time. Interesting!",
                "Just spotted {}. Would you like to explore it?",
            ],
            "unchanged":[
                "Nothing new detected. The environment looks the same.",
                "It seems unchanged. Do you want me to scan again?",
            ],
            "scene_description":[
                "The scene looks like this: {}.",
                "Here’s what I observed: {}.",
            ]
        }

    def add_to_memory(self,obj):
        self.memory.add(obj)

    def generate_response(self,detected_objects,scene_description):
        unseen_objects=[obj for obj in detected_objects if obj not in self.memory]
        
        #updating the memory with new objects
        for obj in unseen_objects:
            self.add_to_memory(obj)

        #avoid repeating same scene description
        if scene_description==self.last_description and not unseen_objects:
            return random.choice(self.templates["unchanged"])
        self.last_description=scene_description

        #dynamic response logic
        if not detected_objects:
            return random.choice(self.templates["unchanged"])

        elif unseen_objects:
            object_list=', '.join(unseen_objects)
            return random.choice(self.templates["discovery"]).format(object_list)

        elif len(detected_objects)>3:
            return f"I see multiple objects: {', '.join(detected_objects)}."

        else:
            object_list=', '.join(detected_objects)
            scene_response=random.choice(self.templates["scene_description"]).format(scene_description)
            return f"{scene_response} Additionally, I noticed {object_list}."

#initialising response generator
response_generator=DynamicResponseGenerator()

#scene description
def generate_scene_description(frame):
    #convert opencv frame (bgr) to pil image (rgb)
    image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    inputs=processor(image,return_tensors="pt")
    out=blip_model.generate(**inputs)
    description=processor.decode(out[0],skip_special_tokens=True)
    return description

#text-to-speech
def describe_scene(description):
    #speaks out the scene description
    print("Description:",description)
    engine.say(description)
    engine.runAndWait()

#main yolo + talk-back pipeline
def scan():
    #initialising camera
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        exit()

    while True:
        ret,frame=cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        #yolo object detection
        results=model(frame)
        detected_objects=[]

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls=int(box.cls[0])
                    object_name=model.names[cls]
                    detected_objects.append(object_name)
        
        #generate scene description
        scene_description=generate_scene_description(frame)
        
        #generate intelligent response
        response=response_generator.generate_response(detected_objects,scene_description)
        
        #speak the response
        describe_scene(response)

        #show yolo output
        frame=results[0].plot()
        cv2.imshow("YOLO + Dynamic Talk-Back",frame)

        #stop on 'q'
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    #release resources
    cap.release()
    cv2.destroyAllWindows()

#run the pipeline
scan()
