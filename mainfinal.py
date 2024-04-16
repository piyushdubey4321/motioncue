import os
import cv2
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import json
from streamlit_lottie import st_lottie


# Function to load lottie file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# Function to recognize gestures from webcam
def recognize_gestures(folder_path, start_detection):
    if not start_detection:
        return

    # Variables
    width, height = 1300, 720

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Get the list of presentation Images
    path_images = sorted(os.listdir(folder_path), key=len)

    # Variables
    img_number = 0
    gesture_threshold = 350
    button_pressed = False
    button_counter = 0
    button_delay = 30
    annotations = [[]]
    annotation_number = 0
    annotation_start = False

    # Hand detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:
        # Import Images
        success, img = cap.read()
        img = cv2.flip(img, 1)
        path_full_image = os.path.join(folder_path, path_images[img_number])
        img_current = cv2.imread(path_full_image)

        hands, img = detector.findHands(img)
        cv2.line(img, (0, gesture_threshold), (width, gesture_threshold), (0, 255, 0), 10)

        if hands and button_pressed is False:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            cx, cy = hand['center']
            lm_list = hand['lmList']

            if cy <= gesture_threshold:
                if cy >= int(np.interp(lm_list[8][0], [height // 2, width], [0, width])):
                    annotation_start = False
                    if fingers == [1, 1, 1, 1, 1]:  # Gesture 1- Left
                        if img_number > 0:
                            button_pressed = True
                            annotations = [[]]
                            annotation_number = 0
                            img_number -= 1

                if cy <= int(np.interp(lm_list[8][0], [height // 2, width], [0, width])):
                    annotation_start = False
                    if fingers == [1, 1, 1, 1, 1]:  # Gesture 2- Right
                        if img_number < len(path_images) - 1:
                            button_pressed = True
                            annotations = [[]]
                            annotation_number = 0
                            img_number += 1

            x_val = int(np.interp(lm_list[8][0], [width // 2, width], [0, width]))
            y_val = int(np.interp(lm_list[8][1], [150, height - 150], [0, height]))
            index_finger = x_val, y_val

            if fingers == [0, 1, 1, 0, 0]:  # Gesture 3 - Show Pointer
                cv2.circle(img_current, index_finger, 12, (0, 0, 255), cv2.FILLED)
                annotation_start = False

            if fingers == [0, 1, 0, 0, 0]:  # Gesture 4- draw Pointer
                if annotation_start is False:
                    annotation_start = True
                    annotation_number += 1
                    annotations.append([])
                cv2.circle(img_current, index_finger, 8, (0, 0, 255), cv2.FILLED)
                annotations[annotation_number].append(index_finger)
            else:
                annotation_start = False

            if fingers == [0, 1, 1, 1, 0]:  # Gesture 5- Erase
                if annotations:
                    if annotation_number >= 0:
                        annotations.pop(-1)
                        annotation_number -= 1
                        button_pressed = True

        # Button Pressed iteration
        if button_pressed:
            button_counter += 1
            if button_counter > button_delay:
                button_counter = 0
                button_pressed = False

        for i in range(len(annotations)):
            for j in range(len(annotations[i])):
                if j != 0:
                    cv2.line(img_current, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 10)

        # Adding webcam image on the slides
        img_small = cv2.resize(img, (213, 120))
        h, w, _ = img_current.shape
        img_current[0:120, w - 213:w] = img_small

        cv2.imshow("Slides", img_current)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # Add a delay between frame updates
        # time.sleep(10)  # Increase the delay to 5 seconds


# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        # Move the uploaded file to the presentation folder
        destination_path = os.path.join("presentation", uploaded_file.name)
        with open(destination_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1, destination_path
    except Exception as e:
        # print(e)
        return 0, None


# Streamlit UI
def main():
    im = "favicon/favicon.ico"
    st.set_page_config(
        page_title="Motion Cue",
        page_icon=im,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # st.title("Motion Cue")
    # Add HTML code with inline CSS to change font size
    st.markdown("""
        <div style='font-size: 100px;font-align: center;
    font-family: cursive; margin-bottom:0%;
    width: 1370.4px;
    margin-left: 420px;
    margin-bottom: -16px;
    margin-top: -57px;
    '>Motion Cue</div>
    """, unsafe_allow_html=True)

    st.markdown("""
            <div style='font-size: 50px;font-align: center;
        font-family: comic sans ms, cursive; 
    margin-left: 292px;
        justify-content: center; color: rgb(0, 0, 0);'>Please Upload Your Presentation</div>
        """, unsafe_allow_html=True)
    # st.header("Gesture Based Navigation")

    lottie_upload = load_lottiefile("lottifilesjson/upload.json")
    st_lottie(lottie_upload, height=200, key="upload")

    # File Uploader
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Display uploaded image
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

            # Save the uploaded file
            status, file_path = save_uploaded_file(uploaded_file)
            if status:
                st.success(f'Image uploaded successfully! File Path: {file_path}')
            else:
                st.error('Failed to upload image!')

    folder_path = "presentation"
    st.write("Press 'q' to exit.")

    # Start detection button
    start_detection = st.button("Start Hand Gesture Detection")

    recognize_gestures(folder_path, start_detection)


if __name__ == "__main__":
    main()


