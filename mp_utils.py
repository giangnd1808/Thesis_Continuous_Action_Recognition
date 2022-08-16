import cv2
import numpy as np
import mediapipe as mp
import time
# import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    global mp_drawing, mp_pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # Draw pose connections

def draw_styled_landmarks(image, results):
    global mp_drawing, mp_pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
def extract_keypoints(results):
    if results.pose_world_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_world_landmarks.landmark]) 

    return pose
