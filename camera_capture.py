# camera_capture.py
import cv2
import numpy as np
from PIL import Image
import argparse

class MathProblemCapturer:
    def __init__(self):
        self.camera = None
        
    def capture_from_webcam(self):
        """Capture math problem using laptop webcam"""
        self.camera = cv2.VideoCapture(0)
        print("Press SPACE to capture, ESC to exit")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            cv2.imshow('Math Problem Capture - Press SPACE to capture', frame)
            
            key = cv2.waitKey(1)
            if key % 256 == 27:  # ESC
                break
            elif key % 256 == 32:  # SPACE
                img_name = "captured_math_problem.png"
                cv2.imwrite(img_name, frame)
                print(f"Image saved as {img_name}")
                break
                
        self.camera.release()
        cv2.destroyAllWindows()
        return img_name
    
    def load_from_file(self, image_path):
        """Load math problem from image file"""
        return image_path