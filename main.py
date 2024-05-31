# # import cvzone
# # import cv2
# # import numpy as np
# # from cvzone.HandTrackingModule import HandDetector
# # import google.generativeai as genai
# # from PIL import Image
# #
# # genai.configure(api_key="AIzaSyBHlut2EcR3PZCGF1qLCFg1G9PkFzaG-ZQ")
# # # The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
# # model = genai.GenerativeModel('gemini-1.5-flash')
# #
# # cap = cv2.VideoCapture(0)
# # cap.set(propId=3, value=1280)
# # cap.set(propId=4, value=720)
# #
# # # Initialize the HandDetector class with the given parameters
# # detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)
# #
# # def getHandinfo(img):
# #     # Find hands in the current frame
# #     # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
# #     # The 'flipType' parameter flips the image, making it easier for some detections
# #     hands, img = detector.findHands(img, draw=True, flipType=True)
# #
# #     # Check if any hands are detected
# #     if hands:
# #         # Information for the first hand detected
# #         hand = hands[0]  # Get the first hand detected
# #         lmList = hand["lmList"]  # List of 21 landmarks for the first hand
# #
# #         # Count the number of fingers up for the first hand
# #         fingers = detector.fingersUp(hand)
# #         print(fingers)
# #         return fingers, lmList
# #     else:
# #         return None
# #
# # def draw(info, prev_pos, canvas):
# #     fingers, lmList = info
# #     current_pos = None
# #
# #     if fingers == [0, 1, 0, 0, 0]:
# #         current_pos = lmList[8][0:2]
# #         if prev_pos is None: prev_pos = current_pos
# #         cv2.line(canvas, prev_pos, current_pos, color=(255, 0, 255), thickness=10)
# #
# #     elif fingers == [0,0,1,0,0]:
# #         canvas = np.zeros_like(img)
# #
# #         return current_pos, canvas
# #
# # def sendToAi(model,canvas,fingers):
# #     if fingers == [1,0,0,0,1]:
# #         pil_image = Image.fromarray(canvas)
# #         response = model.generate_content(["solve this math problem",pil_image])
# #         # response = model.generate_content("who ia prime minister of india")
# #         print(response.text)
# # prev_pos = None
# # canvas = None
# #
# # # Continuously get frames from the webcam
# # while True:
# #     # Capture each frame from the webcam
# #     # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
# #     success, img = cap.read()
# #     img = cv2.flip(img, flipCode=1)
# #
# #     if canvas is None:
# #         canvas = np.zeros_like(img)
# #
# #     info = getHandinfo(img)
# #     if info:
# #         fingers, lmList = info
# #         print(fingers)
# #         prev_pos, canvas = draw(info, prev_pos, canvas)
# #         sendToAi(model,canvas,fingers)
# #     # else:
# #     #     prev_pos, canvas = prev_pos, canvas
# #
# #     image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
# #
# #     # Display the image in a window
# #     cv2.imshow("Image", img)
# #     cv2.imshow("Canvas", canvas)
# #     cv2.imshow("Image Combined", image_combined)
# #
# #     # Keep the window open and update it for each frame; wait for 1 millisecond between frames
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # cap.release()
# # cv2.destroyAllWindows()
# import cvzone
# import cv2
# import numpy as np
# from cvzone.HandTrackingModule import HandDetector
# import google.generativeai as genai
# from PIL import Image
# import streamlit as st
#
# st.set_page_config(layout="wide")
# st.image('img.png')
#
# col1, col2 = st.columns([2, 1])
# with col1:
#     run = st.checkbox('Run', value=True)
#     FRAME
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Streamlit configuration
st.set_page_config(layout="wide")
st.image('img.png')

col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")

# Configure the Google Generative AI model
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the webcam opened successfully
if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    st.stop()

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandinfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, color=(255, 0, 255), thickness=10)
        prev_pos = current_pos

    elif fingers == [0, 0, 1, 0, 0]:
        canvas = np.zeros_like(canvas)
        prev_pos = None

    return prev_pos, canvas

def sendToAi(model, canvas, fingers):
    if fingers == [1, 0, 0, 0, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["solve this math problem", pil_image])
        return response.text
    return None

prev_pos = None
canvas = None
image_combined = None
output_text = ""

# Continuously get frames from the webcam
while run:
    success, img = cap.read()
    if not success:
        st.warning("Warning: Could not read frame from webcam.")
        continue

    img = cv2.flip(img, flipCode=1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandinfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAi(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
