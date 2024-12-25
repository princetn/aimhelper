import cv2
import pyautogui
from ultralytics import YOLO
import time
import mss
import imutils
import numpy as np
from pynput import mouse, keyboard
import pythoncom
import pyWinhook
import threading
import math


radius = 600
# Load the model
yolo = YOLO('yolov8s.pt',verbose=False)


# Load the video capture
videoCap = cv2.VideoCapture(0)
left = False
right = False
quit = False
def OnMouseEvent():
    global left
    global right
    with mouse.Events() as mouseEvents:
        mouseEvent = mouseEvents.get(0.02)
        if mouseEvent != None:
            #print(mouseEvent)
            if "Button.left" in str(mouseEvent):
                if "pressed=True" in str(mouseEvent):
                    left = not left
                elif "pressed=False" in str(mouseEvent):
                    left = False
            elif "Button.right" in str(mouseEvent):
                if "pressed=True" in str(mouseEvent):
                    right = True
                elif "pressed=False" in str(mouseEvent):
                    right = False


def OnKeyBoardEvent():
    with keyboard.Events() as key:
        keyEvent = key.get(0.02)
        if keyEvent != None:
            print(str(keyEvent))
            if("Key.delete" in str(keyEvent)):
                return True
            else:
                return False




def EventsThred():
    global quit
    while True:
        OnMouseEvent()
        if OnKeyBoardEvent():
            break
        #time.sleep(0.02)
    quit = True


thread = threading.Thread(target=EventsThred)
thread.start()
# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

img = None
sct = mss.mss()
monitor_number = 1
mon = sct.monitors[monitor_number]

# The screen part to capture
monitor = {
    "top": mon["top"],
    "left": mon["left"],
    "width": mon["width"],
    "height": mon["height"],
    "mon": monitor_number,
}

while True:
    #ret, frame = videoCap.read()
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    results = yolo.track(frame, stream=True, verbose=False)

    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.7 and box.cls[0] == 0:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x, y = pyautogui.position()
                d = math.sqrt(math.pow(x-(x1+x2)/2.0,2)+math.pow(y-(y1+10),2))
                print("distance: ", d)
                if d <= radius:
                    if True:
                        pyautogui.moveTo((x1+x2)/2, y1+100)


                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # get the respective colour
                colour = getColours(cls)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                # cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    # show the image
    cv2.imshow('frame', frame)

    # break the loop if 'q' is pressed
    cv2.waitKey(1)
    if quit:
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()