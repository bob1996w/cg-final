import cv2
import imageProcessing as ip
import sys
try:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
except:
    print('usage:')
    print('python imgtest.py input_file, output_file')

img = cv2.imread(input_file)
# img = cv2.imread('../landscape.jpg')
img = cv2.resize(img, (640, 480))
p = ip.painterly(img, [8, 4, 2])
# cv2.imshow('p', p)
cv2.imwrite(output_file, p)
# cv2.waitKey(0)
# cv2.destroyAllWindows()