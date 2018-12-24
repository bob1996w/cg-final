import PIL
from PIL import Image,ImageTk
# import pytesseract
import cv2
import tkinter as tk
import time

import strings as s
import imageProcessing as ip

# CV setup
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# global variables
gCameraEffect = False
gCurrentFrame = None

# build screen
root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())


# callbacks
def toggleCameraEffect():
    global gCameraEffect
    gCameraEffect = not gCameraEffect
    svMessage.set(s.MESSAGE[1] if gCameraEffect else s.MESSAGE[0])
def previewWindow():
    global gCurrentFrame
    frame = ip.painterly(gCurrentFrame, [6, 4, 2])
    window = tk.Toplevel(root)
    lTransformed = tk.Label(window)
    lTransformed.pack()

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lTransformed.imgtk = imgtk
    lTransformed.configure(image=imgtk)


# upper frame
fMain = tk.Frame(root)
fMain.pack()
lmain = tk.Label(root)
lmain.pack()
# lower frame
fLower = tk.Frame(root)
fLower.pack(side=tk.BOTTOM, fill=tk.X)
svMessage = tk.StringVar()
svMessage.set(s.MESSAGE[1] if gCameraEffect else s.MESSAGE[0])
lDescription = tk.Label(fLower, textvariable=svMessage, fg='black')
lDescription.pack(anchor=tk.W, side=tk.LEFT, expand=True)
bToggleCameraEffect = tk.Button(fLower, text='toggle effect', fg='black', command=toggleCameraEffect)
bToggleCameraEffect.pack(anchor=tk.E, side=tk.LEFT, expand=False)
bScreenShot = tk.Button(fLower, text='screenshot&apply', fg='black', command=previewWindow)
bScreenShot.pack(anchor=tk.E, side=tk.RIGHT, expand=False)

frameSize = cap.read()[1].shape
print('frame size:', frameSize)
croppedFrameSize = (frameSize[0] // 4, frameSize[0] * 3 // 4, frameSize[1] // 4, frameSize[1] * 3 // 4)

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)[croppedFrameSize[0]:croppedFrameSize[1], croppedFrameSize[2]:croppedFrameSize[3]]
    global gCurrentFrame
    gCurrentFrame = frame

    global gCameraEffect
    if gCameraEffect:
    # if True:
        # frame = ip.blackAndWhite(frame)
        frame = ip.painterly(frame, [8, 4, 2])

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()