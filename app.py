import streamlit as st
import numpy as np
import tensorflow as tf 
import cv2
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import MeanSquaredError
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pygame  


MODEL_PATH = "fabric_quality_model.h5"
autoencoder = load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()})


pygame.mixer.init()
ALARM_SOUND = "alarm.mp3"

NUM_CAMERAS = 10
cameras = {
    "Machine 1": cv2.VideoCapture("rtsp://username:password@192.168.1.101:554/stream"),
    "Machine 2": cv2.VideoCapture("rtsp://username:password@192.168.1.102:554/stream"),
    "Machine 3": cv2.VideoCapture("rtsp://username:password@192.168.1.103:554/stream"),
    "Machine 4": cv2.VideoCapture("rtsp://username:password@192.168.1.104:554/stream"),
    "Machine 5": cv2.VideoCapture("rtsp://username:password@192.168.1.105:554/stream"),
    "Machine 6": cv2.VideoCapture("rtsp://username:password@192.168.1.106:554/stream"),
    "Machine 7": cv2.VideoCapture("rtsp://username:password@192.168.1.107:554/stream"),
    "Machine 8": cv2.VideoCapture("rtsp://username:password@192.168.1.108:554/stream"),
    "Machine 9": cv2.VideoCapture("rtsp://username:password@192.168.1.109:554/stream"),
    "Machine 10":cv2.VideoCapture("rtsp://username:password@192.168.1.110:554/stream")
}


for machine, cam in cameras.items():
    if not cam.isOpened():
        st.warning(f" {machine} camera not detected! Using laptop camera instead.")
        cameras[machine] = cv2.VideoCapture(0) 


reference_images = {}
monitoring_status = {f"Machine {i+1}": False for i in range(NUM_CAMERAS)}


def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img


def compute_mse(img1, img2):
    return np.mean(np.square(img1 - img2))


def compute_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(img1_gray, img2_gray)


def capture_image(camera):
    if camera.isOpened():
        ret, frame = camera.read()
        if ret:
            return frame
    return None


def start_alarm(machine):
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(ALARM_SOUND)
        pygame.mixer.music.play(-1)
    st.error(f" BAD FABRIC DETECTED in {machine}! ALARM ACTIVATED!")


def stop_alarm():
    pygame.mixer.music.stop()


st.title("Fabric Quality Monitoring System")
st.write("Monitoring fabric quality across 10 machines with real-time analysis.")


col1, col2 = st.columns(2)
start_all = col1.button("Start Monitoring for All Machines")
stop_all = col2.button("Stop Monitoring for All Machines")


if start_all:
    for machine, camera in cameras.items():
        st.write(f"Capturing Reference Fabric Image for {machine}...")
        reference_image = capture_image(camera)
        if reference_image is not None:
            reference_images[machine] = reference_image
            monitoring_status[machine] = True
            st.success(f"Monitoring started for {machine}")
        else:
            st.error(f"Camera not detected for {machine}. Check the connection.")


if stop_all:
    for machine in monitoring_status:
        monitoring_status[machine] = False
    stop_alarm()
    st.success("Monitoring Stopped for All Machines.")


for machine, camera in cameras.items():
    if monitoring_status[machine]:
        with st.expander(f"Live Monitoring: {machine}", expanded=True):
            st.write(f"Analyzing fabric quality for {machine}...")

            
            sample_image = capture_image(camera)
            if sample_image is not None:
                st.image(sample_image, caption=f"Live Image from {machine}", channels="BGR", use_column_width=True)

                
                ref_img = reference_images[machine]
                ref_features = autoencoder.predict(preprocess_image(ref_img))
                sample_features = autoencoder.predict(preprocess_image(sample_image))

                
                mse_value = compute_mse(ref_features, sample_features)
                ssim_value = compute_ssim(ref_img, sample_image)

                
                threshold_mse_good = 0.0125
                threshold_mse_bad = 0.0180
                threshold_ssim_good = 0.88
                threshold_ssim_bad = 0.75

                
                if mse_value < threshold_mse_good and ssim_value > threshold_ssim_good:
                    result = "Good Fabric"
                    st.success(f" {machine}: Fabric quality is good.")
                    stop_alarm()
                elif mse_value > threshold_mse_bad or ssim_value < threshold_ssim_bad:
                    result = "Bad Fabric"
                    start_alarm(machine)
                else:
                    result = "Borderline Quality"
                    st.warning(f" {machine}: Fabric is borderline. Further inspection needed.")

                
                st.subheader(f"Prediction: {result}")
                st.write(f" **MSE Score**: {mse_value:.10f}")
                st.write(f" **SSIM Score**: {ssim_value:.4f}")

            else:
                st.error(f"Failed to capture image for {machine}. Check the camera connection.")

        time.sleep(1)  
