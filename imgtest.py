import cv2
import imageProcessing as ip

img = cv2.imread('../DSC_7535.JPG')
img = cv2.resize(img, (640, 480))
p = ip.painterly(img, [10, 5])
# cv2.imshow('p', p)
cv2.imwrite('../image.png', p)
# cv2.waitKey(0)
# cv2.destroyAllWindows()