from ultralytics import YOLO
import streamlit as st
import numpy as np 
import cv2
import os 

model_path = os.path.join('.', 'custom_training', 'runs', 'detect', 'train3', 'weights', 'best.pt')

# Set title
st.title('Vessel Detection Using Python')

# Set header
st.header('Please Upload your satellite image')

# File upload
file = st.file_uploader('', type=['tif', 'jpg'])

# Display image
if file is not None:
    image = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(image, 1)
    st.image(opencv_image, channels="BGR", use_column_width=True)

    # Load model
    model = YOLO(model_path)  # Loading the custom model

    # Perform multi-scale detection using sliding window
    window_size = (416, 416)  # Size of the sliding window
    stride = 100  # Stride for moving the sliding window

    detections = []

    for y in range(0, opencv_image.shape[0] - window_size[1], stride):
        for x in range(0, opencv_image.shape[1] - window_size[0], stride):
            window = opencv_image[y:y + window_size[1], x:x + window_size[0]]

            # Perform inference on the window
            results = model(window)

            # Process detections
            for detection in results.xyxy[0]:  # Get detections from the first image in the batch
                class_id, confidence, xmin, ymin, xmax, ymax = detection.tolist()

                # Translate bounding box coordinates to original image coordinates
                xmin += x
                ymin += y
                xmax += x
                ymax += y
                
                detections.append([class_id, confidence, xmin, ymin, xmax, ymax])

    # Draw bounding boxes on the original image
    for class_id, confidence, xmin, ymin, xmax, ymax in detections:
        # Filter out detections with low confidence
        if confidence > 0.3:
            # Draw bounding box on the image
            cv2.rectangle(opencv_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    # Display the result image
    st.image(opencv_image, channels="BGR", use_column_width=True)
