import cv2
import numpy as np
import random as rnd

# img = cv2.imread('../test1.jpg')
img = np.zeros((200, 200, 3), np.uint8)
for i in range(10):
    y = rnd.randint(0, img.shape[0] - 16 - 1)
    x = rnd.randint(0, img.shape[1] - 16 - 1)
    print(y, x)
    overlay_base = img[y: y + 16, x: x + 16]
    overlay = cv2.circle(overlay_base, (8, 8), 8, (255, 255, 255), -1)
    # overlay = cv2.GaussianBlur(overlay, (0, 0), 4)
    img[y: y + 16, x: x + 16] = overlay
    cv2.imshow(str(i), overlay)
    # overlay = cv2.circle(overlay_base, (100, 100), 5, (0, 255, 0), -1)
    # img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
cv2.imshow('h', img)
cv2.waitKey(0)