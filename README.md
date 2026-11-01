# YOLOv8-Laptop-Camera-Object-Detection
# Name: Sanchita Sandeep
# Reg No: 212224240142
## Aim:
To perform real-time object detection on a live video stream captured from a webcam using the YOLO (You Only Look Once) deep learning model and OpenCV.
## Steps:
Open Jupyter Notebook using Anaconda.
Access your laptop camera using OpenCV.
Wait for 5 seconds and capture an image.
Display the captured image.
Apply YOLOv8 object detection.
Display the detected image with bounding boxes and labels.
## Algorithm:
```

Laptop Camera
      ↓
Capture Image
      ↓
Display Image
      ↓
YOLOv8 Detection
      ↓
Display Detected Objects
```
## Program:
```
!pip install ultralytics opencv-python matplotlib
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
model = YOLO("yolov8m.pt")
import cv2
from ultralytics import YOLO
from IPython.display import display, clear_output, Image
import PIL.Image
import io

# Load model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    try:
        print("Streaming... Press the Stop (■) button on the toolbar to end.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break 
            
            # Perform object detection
            results = model(frame, conf=0.60, verbose=False)
            annotated_frame = results[0].plot()
            
            # Convert OpenCV BGR to RGB, then to PIL Image
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = PIL.Image.fromarray(annotated_frame)
            
            # Encode to JPEG bytes to display smoothly in Jupyter
            f = io.BytesIO()
            img_pil.save(f, format="JPEG")
            
            # Clear output and display the new frame instantly
            clear_output(wait=True)
            display(Image(data=f.getvalue()))
            
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        print("Webcam released.")



```

## Output:
Object detected
<img width="640" height="480" alt="download" src="https://github.com/user-attachments/assets/87eab281-2ff3-4322-96a9-929139c4ee15" />
## Result
Object Detected Successfully using YOLO in webcam 






