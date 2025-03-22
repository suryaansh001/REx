import pyttsx3
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak_objects(detected_objects, last_spoken):
    
    
    if not detected_objects:
        if last_spoken!=[]:
            engine.say("Nothing detected")
            engine.runAndWait()
        return []
    
    object_names = [obj for obj in detected_objects]
    if object_names != last_spoken:
        engine.say("I see " + ", ".join(object_names))
        engine.runAndWait()
    return object_names