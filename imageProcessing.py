import cv2
import numpy as np

def blackAndWhite(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = np.uint8((int(img[i, j, 0]) + int(img[i, j, 1]) + int(img[i, j, 2])) // 3)
            img[i, j, 0] = val
            img[i, j, 1] = val
            img[i, j, 2] = val
    return img

def cv2_blackAndWhite(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aaron Hertzmann, 
# Painterly Rendering with Curved Brush Strokes of Multiple Sizes

# CONSTANTS
GAUSSIAN_BLUR_KERNEL = (0, 0)
BLUR_FACTOR = 1
POST_BLUR_FACTOR = 0.1
GRID_SIZE = 1
APPROXIMATION_THRESHOLD = 100

def painterly(img, radii=[2]):
    """
    Paint the original img with multiple paint brushes-style.

    :param np.ndarray img: The original image. np.uint8 format.
    :param float radii: A list consisting of different brush radii.
    :return np.ndarray: The output image. np.uint8 format.
    """
    # all use uint8
    canvas = np.ones(img.shape, np.uint8) * 255
    radii.sort(reverse=True)
    for radius in radii:
        print('paint radius', radius)
        refImage = cv2.GaussianBlur(img, GAUSSIAN_BLUR_KERNEL, BLUR_FACTOR * radius)
        canvas = paintLayer(canvas, refImage, radius, img)
        canvas = cv2.GaussianBlur(canvas, GAUSSIAN_BLUR_KERNEL, POST_BLUR_FACTOR * radius)
    return canvas

def argmax2D(X):
    """
    Input a ndarray, output the max's position.

    :param arr: ndarray
    """
    return np.unravel_index(X.argmax(), X.shape)

def paintToCanvasWithRef(canvas, stroke):
    """
    Paint a circle to canvas by stroke.
    """
    x = stroke['x']
    y = stroke['y']
    refImage = stroke['referenceImage']
    r = stroke['R']
    color = (int(refImage[y,x,0]), int(refImage[y,x,1]), int(refImage[y,x,2]))
    # print(type(refImage[y,x,0]))
    canvas = cv2.circle(canvas, (x,y), r, color, -1)
    return canvas

def paintToCanvasWithSource(canvas, stroke, source):
    """
    Paint a circle to canvas by stroke, and by using the color from the source image.
    """
    x = stroke['x']
    y = stroke['y']
    r = stroke['R']
    color = (int(source[y,x,0]), int(source[y,x,1]), int(source[y,x,2]))
    canvas = cv2.circle(canvas, (x,y), r, color, -1)
    return canvas



def paintLayer(canvas, refImage, radius, source):
    """
    Paint the layer with single brush radius.

    :param np.ndarray canvas: The current canvas. np.uint8 format.
    :param np.ndarray refImage: The reference image for this layer. np.uint8 format.
    :param float radius: Current brush size.
    :param np.ndarray source: The source image. np.uint8 format.
    :return np.ndarray: Canvas after painting. np.uint8 format.
    """
    strokes = []
    # use float16
    # pointwise difference image
    diff_3 = np.abs(np.float32(canvas) - np.float32(refImage))
    # print('diff_3 dtype', diff_3.dtype)
    diff = np.sqrt(diff_3[:,:,0] ** 2 + diff_3[:,:,1] ** 2+ diff_3[:,:,2] ** 2)
    # print(diff.dtype)
    grid = GRID_SIZE * radius
    for i in range(0, canvas.shape[0] + grid // 2, grid):
        for j in range(0, canvas.shape[1] + grid // 2, grid):
            # sum the error near (j, i)
            Ystart = 0 if int(i - grid / 2) <= 0 else int(i - grid / 2)
            Yend = canvas.shape[0] - 1 if int(i + grid / 2) >= canvas.shape[0] - 1 else int(i + grid / 2)
            Xstart = 0 if int(j - grid / 2) <= 0 else int(j - grid / 2)
            Xend = canvas.shape[1] - 1 if int(j + grid / 2) >= canvas.shape[1] - 1 else int(j + grid / 2)
            realGridSize = (Yend - Ystart) * (Xend - Xstart) if (Yend - Ystart) * (Xend - Xstart) > 0 else 1
            if realGridSize <= 0:
                print('realGridSize error at', radius, i, j)

            # area = diff[int(i - grid/2):int(i + grid/2), int(j-grid/2):int(j+grid/2)]
            # areaError = np.sum(area) / (grid ** 2)
            area = diff[Ystart:Yend, Xstart:Xend]
            areaError = np.sum(area) / realGridSize
            # print('areaError', areaError)
            if areaError > APPROXIMATION_THRESHOLD:
                # find the largest error point
                maxPoint = argmax2D(area)
                # print(maxPoint)
                # make stroke
                strokes.append({'R': radius, 
                    'x': int(j - grid / 2) + maxPoint[1], 
                    'y': int(i - grid / 2) + maxPoint[0], 
                    'referenceImage': refImage})

    # paint all stroke in S on the canvas, in random order
    order = np.random.permutation(len(strokes))
    print(len(strokes))
    # currently paint in circles
    for o in order:
        # canvas = paintToCanvas(canvas, strokes[o])
        canvas = paintToCanvasWithSource(canvas, strokes[o], source)

    return canvas