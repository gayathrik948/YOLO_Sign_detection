from ultralytics import YOLO
import cv2
from PIL import Image
import streamlit as st


def image_stream(image_path):
    model = YOLO('best.pt')
    results = model(image_path)

    processed_images = []
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
        processed_images.append(im)

    return processed_images


def video_stream(video_name):
    cap = cv2.VideoCapture(video_name)
    model = YOLO('best.pt')

    stframe = st.empty()  # Streamlit element to display video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        res_plotted = results[0].plot()

        # Convert the OpenCV BGR image to RGB
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        # Display the processed frame using stframe
        stframe.image(res_plotted_rgb, channels='RGB')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


