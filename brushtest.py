import cv2
import numpy as np

brush = cv2.imread('brush.png', cv2.IMREAD_UNCHANGED)
brush_shape = brush.shape
print(brush_shape)
img = np.zeros((200, 200, 4), np.uint8)
img[:, :, 3] = np.ones((200, 200), np.uint8)

# img_overlay = cv2.addWeighted(img[100:100 + brush_shape[0], 100:100+brush_shape[1]].copy(), 0.5, brush, 0.5, 0)
alpha_s = brush[:, :, 3]/ 255.0
alpha_l = 1 - alpha_s
for i in range(3):
    img[100:100+brush_shape[0], 100:100+brush_shape[1], i] = alpha_s * brush[:, :, i] + alpha_l * img[100:100+brush_shape[0], 100:100+brush_shape[1], i]
cv2.imshow('a', img)
cv2.waitKey(0)