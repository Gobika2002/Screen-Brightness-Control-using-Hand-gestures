import cv2
import mediapipe as mp
import numpy as np  
import screen_brightness_control as sbc  # Import the screen brightness control module
from math import hypot  # Calculate the distance between two points

cap = cv2.VideoCapture(0)  # Open the webcam

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils # drawing_utils for drawing hand landmarks in media pipe

while True:
    a, image = cap.read()  # Read a frame from the webcam
    if not a or image is None:
        continue  # Skip if frame not captured

    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
    results = hands.process(imgRGB)  # Process the RGB image to find hands
    lmList = []  # Initialize an empty list to store landmark positions

    if results.multi_hand_landmarks:  # If hands are found
        for handLms in results.multi_hand_landmarks:
            h, w = image.shape[:2]  # Use only height and width
            for id, lm in enumerate(handLms.landmark): # id is the index of the landmark
                cx, cy = int(lm.x * w), int(lm.y * h) # lm is a landmark object with x and y attributes
                lmList.append([id, cx, cy]) #append function adds the landmark id and its coordinates to the list
            mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
            # print(lmList)  # Optionally print the list of landmarks

    if len(lmList) > 8:
        try:
            # Calculate the distance between the thumb tip and index finger tip
            x1, y1 = lmList[4][1], lmList[4][2] # lmList[4] is the thumb tip, lm[4][1] is the x coordinate of the thumb tip, lm[4][2] is the y coordinate of the thumb tip
            x2, y2 = lmList[8][1], lmList[8][2] #lmList[8] is the index finger tip, lm[8][1] is the x coordinate of the index finger tip, lm[8][2] is the y coordinate of the index finger tip
            cv2.circle(image, (x1, y1), 10, (255, 0, 0), cv2.FILLED) # Draw a filled circle at the thumb tip
            cv2.circle(image, (x2, y2), 10, (255, 0, 0), cv2.FILLED) # Draw a filled circle at the index finger tip
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            length = hypot(x2 - x1, y2 - y1) # Calculate the distance between the thumb and index finger tips
            # Map the length to a brightness value between 0 and 100
            bright = np.interp(length, [15, 220], [0, 100])
            print(bright, length)
            sbc.set_brightness(int(bright))
        except Exception as e:
            print("Landmark error:", e) # Handle any exceptions that occur during landmark processing

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently.
# The `hypot` function is used to calculate the Euclidean distance between two points, which is useful for determining the distance between the thumb and index finger tips.
# The `np.interp` function is used to map the length of the line between the thumb and index finger to a brightness value between 0 and 100.
# The `sbc.set_brightness` function is used to set the screen brightness based on the calculated value.
# The `cv2.imshow` function is used to display the image with the drawn landmarks and lines.
# The 'cv2.circle' function is used to draw circles at the thumb and index finger tips, and the 'cv2.line' function is used to draw a line between them.
# The 'cv2.filled' parameter in the 'cv2.circle' function is used to fill the circle with color.