# from ultralytics import YOLO


# model_path = pathlib.Path(__file__).parent / "models" / "best.pt"

# model = YOLO(model_path)

########
model_no = 2
use_webcam = False
video_link = "C:\\Users\\Ahmad\\Downloads\\Sample 2 smaller res.mp4"

######

import cv2
import numpy as np
import pathlib
from roboflow import Roboflow

rf = Roboflow(api_key="0I8SBTJUGecS1uO3wyrn")

if model_no == 1:
    project = rf.workspace().project("construction-site-safety")
    model = project.version(27).model
elif model_no == 2:
    project = rf.workspace().project("hard-hats-fhbh5")
    model = project.version(4).model
else:
    project = rf.workspace().project("safety-helmet-dataset-uvh1t")
    model = project.version(1).model

if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_link)
    
    
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference on the video stream
    result = model.predict(frame_rgb, confidence=40, overlap=30).json()

    predictions = result["predictions"]

    # Draw bounding boxes and labels on the frame
    for pred in predictions:
        x, y, width, height, confidence, label = (
            pred["x"],
            pred["y"],
            pred["width"],
            pred["height"],
            pred["confidence"],
            pred["class"],
        )
        if confidence > 0.5:  # You can adjust the threshold
            x1, y1 = int(x - width / 2), int(y - height / 2)
            x2, y2 = int(x + width / 2), int(y + height / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label}: {confidence:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# infer on a local image
# file_name = "two-young-construction-workers-wearing-555864.jpg"
# image_path = str(pathlib.Path(__file__).parent / "source_files" / file_name)
# print(model.predict(image_path, confidence=40, overlap=30).json())

# visualize your prediction
# model.predict(image_path, confidence=40, overlap=30).save("prediction.jpg")
# print(model)
# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
