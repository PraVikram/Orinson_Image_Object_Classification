import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

model = YOLO("yolov8n.pt")

st.title("Object Detection and Classification")
st.write("Upload an image to detect and classify objects using YOLOv8.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Run YOLOv8 model on the image
    results = model(image_np)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the result to a format suitable for drawing
    annotated_image = image_np.copy()
    detected_objects = []

    # Iterates through detections and draw bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()  # Extract confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Extract class IDs as integers

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[cls_id]
            detected_objects.append(class_name)

            # Draw rectangle and put text on the image
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(annotated_image, caption='Detected Objects', use_column_width=True)

    # Display detected object names
    if detected_objects:
        st.write("Detected objects in the image:")
        st.write(", ".join(set(detected_objects)))
    else:
        st.write("No objects detected in the image.")

else:
    st.warning("Please upload an image.")
