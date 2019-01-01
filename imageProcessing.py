import cv2
import numpy as np
import math

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

SOBEL_KERNEL_X = np.array([[1,0,-1], [2, 0, -2], [1, 0, -1]])
SOBEL_KERNEL_Y = SOBEL_KERNEL_X.T

# CONSTANTS
GAUSSIAN_BLUR_KERNEL = (0, 0)
BLUR_FACTOR = 0.7
# POST_BLUR_FACTOR = 0.2
GRID_SIZE = 1
APPROXIMATION_THRESHOLD = 100
MIN_STROKE_LENGTH = 4
MAX_STROKE_LENGTH = 16

CURVATURE_FILTER = 1

JITTER_HUE = 0.  # range [0, 180]
JITTER_SAT = 0.  # range [0, 255]
JITTER_VAL = 0.  # range [0, 255]
JITTER_R   = 0.
JITTER_G   = 0.
JITTER_B   = 0.

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
        # canvas = cv2.GaussianBlur(canvas, GAUSSIAN_BLUR_KERNEL, POST_BLUR_FACTOR * radius)
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

def paintSplineStrokesToCanvas(canvas, stroke, source):
    """
    Paint spline strokes to canvas.
    """
    points = stroke['p']
    color = stroke['c']
    r = stroke['R']
    for p in points:
        canvas = cv2.circle(canvas, (p[0], p[1]), r, color, -1, cv2.LINE_AA)
    return canvas


def colorAbs(colorA, colorB):
    diff = np.float32(colorA) - np.float32(colorB)
    return np.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)

def luminance(img):
    return 0.11 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.30 * img[:, :, 2]

def pictureGradient(img, x, y):
    """
    Calculate picture gradient, return (gx, gy)
    """
    gx = img[y, x+1] - img[y, x-1]
    gy = img[y+1, x] - img[y-1, x]
    return (gx, gy)

def jitterColor(colorList):
    """
    Add jitter to color.
    :param np.ndarray color: Color to jitter. np.uint8. BGR format.
    :return np.ndarray: jittered color. BGR format.
    """
    # TODO: implement HSV, RGB jitters
    color_HSV = cv2.cvtColor(np.uint8([[colorList]]), cv2.COLOR_BGR2HSV)[0, 0]
    # decide by gaussian distribution
    color_HSV[0] = np.uint8(np.clip(np.random.normal(color_HSV[0], 180 * JITTER_HUE), 0, 180))
    color_HSV[1] = np.uint8(np.clip(np.random.normal(color_HSV[1], 256 * JITTER_SAT), 0, 255))
    color_HSV[2] = np.uint8(np.clip(np.random.normal(color_HSV[2], 256 * JITTER_VAL), 0, 255))
    color = cv2.cvtColor(np.uint8([[color_HSV]]), cv2.COLOR_HSV2BGR)[0, 0]
    color[0] = np.uint8(np.clip(np.random.normal(color[0], 256 * JITTER_B), 0, 255))
    color[1] = np.uint8(np.clip(np.random.normal(color[1], 256 * JITTER_G), 0, 255))
    color[2] = np.uint8(np.clip(np.random.normal(color[2], 256 * JITTER_R), 0, 255))
    return color

def makeSplineStroke_broken(canvas, stroke):
    """
    Paint a spline stroke on canvas.
    """
    x0 = stroke['x']
    y0 = stroke['y']
    refImage = stroke['referenceImage']
    r = stroke['R']

    strokeColor = refImage[y0, x0]
    strokeColorA = (int(strokeColor[0]), int(strokeColor[1]), int(strokeColor[2]))
    Ks = []
    Ks.append([x0, y0])
    (x, y) = (x0, y0)
    (lastDx, lastDy) = (0, 0)
    for i in range(1, MAX_STROKE_LENGTH + 1):
        if y >= refImage.shape[0] or x >= refImage.shape[1] or y < 0 or x < 0:
            return (Ks, strokeColorA)
        # print ('stroke length', i)
        if i > MIN_STROKE_LENGTH and colorAbs(refImage[y, x], canvas[y, x]) < colorAbs(refImage[y, x], strokeColor):
            return (Ks, strokeColorA)
        
        lumImage = luminance(refImage) [y - 1:y + 2, x - 1:x + 2]
        # gx = cv2.Sobel(lumImage, cv2.CV_16S, 1, 0)[y, x]
        # gy = cv2.Sobel(lumImage, cv2.CV_16S, 0, 1)[y, x]
        # gx = cv2.Sobel(lumImage[y-1:y+2, x-1:x+2], cv2.CV_16S, 1, 0, ksize=3)[0,0]
        # gy = cv2.Sobel(lumImage[y-1:y+2, x-1:x+2], cv2.CV_16S, 0, 1, ksize=3)[0,0]
        if y-1 < 0 or y+1 >= refImage.shape[0] or x-1 < 0 or x+1 >= refImage.shape[1]:
            gx = 0
            gy = 0
        else:
            gy = np.multiply(lumImage, SOBEL_KERNEL_Y)[0,0]
            gx = np.multiply(lumImage, SOBEL_KERNEL_X)[0,0]
        
        # detect vanishing gradient
        gMag = math.sqrt(gx ** 2 + gy ** 2)
        if gMag < 0.0001:
            return (Ks, strokeColorA)
        
        # get unit vector of gradient
        gx /= gMag
        gy /= gMag
        # compute a normal direction
        (dx, dy) = (-gy, gx)

        # if necessary, reverse the direction
        if lastDx * dx + lastDy * dy < 0:
            (dx, dy) = (-dx, -dy)
        
        # filter the stroke direction
        (dx, dy) = CURVATURE_FILTER * (dx, dy) + (1-CURVATURE_FILTER) * (lastDx, lastDy)
        dd = np.sqrt(dx ** 2 + dy ** 2)
        dx /= dd
        dy /= dd
        x = int(x + r * dx)
        y = int(y + r * dy)
        (lastDx, lastDy) = (dx, dy)

        Ks.append([x, y])
    return (Ks, strokeColorA)



def makeSplineStroke(canvas, stroke):
    """
    Paint a spline stroke on canvas.
    """
    x0 = stroke['x']
    y0 = stroke['y']
    refImage = stroke['referenceImage']
    r = stroke['R']

    strokeColor = refImage[y0, x0]
    strokeColor = jitterColor(strokeColor)
    strokeColorA = (int(strokeColor[0]), int(strokeColor[1]), int(strokeColor[2]))
    Ks = []
    Ks.append([x0, y0])
    (x, y) = (x0, y0)
    (lastDx, lastDy) = (0, 0)
    for i in range(1, MAX_STROKE_LENGTH + 1):
        if y >= refImage.shape[0] or x >= refImage.shape[1] or y < 0 or x < 0:
            return (Ks, strokeColorA)
        # print ('stroke length', i)
        if i > MIN_STROKE_LENGTH and colorAbs(refImage[y, x], canvas[y, x]) < colorAbs(refImage[y, x], strokeColor):
            return (Ks, strokeColorA)
        
        lumImage = luminance(refImage)
        gx = cv2.Sobel(lumImage, cv2.CV_16S, 1, 0)[y, x]
        gy = cv2.Sobel(lumImage, cv2.CV_16S, 0, 1)[y, x]
        
        # detect vanishing gradient
        gMag = math.sqrt(gx ** 2 + gy ** 2)
        if gMag < 1e-6:
            return (Ks, strokeColorA)
        
        # get unit vector of gradient
        gx /= gMag
        gy /= gMag
        # compute a normal direction
        (dx, dy) = (-gy, gx)

        # if necessary, reverse the direction
        if lastDx * dx + lastDy * dy < 0:
            (dx, dy) = (-dx, -dy)
        
        # filter the stroke direction
        dx = CURVATURE_FILTER * dx + (1 - CURVATURE_FILTER) * lastDx
        dy = CURVATURE_FILTER * dy + (1 - CURVATURE_FILTER) * lastDy
        dd = np.sqrt(dx ** 2 + dy ** 2)
        dx /= dd
        dy /= dd
        x = int(x + r * dx)
        y = int(y + r * dy)
        (lastDx, lastDy) = (dx, dy)

        Ks.append([x, y])
    return (Ks, strokeColorA)




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
                # strokes.append({'R': radius, 
                #     'x': int(j - grid / 2) + maxPoint[1], 
                #     'y': int(i - grid / 2) + maxPoint[0], 
                #     'referenceImage': refImage})
                initStroke = {'R': radius, 'x': int(j - grid/2) + maxPoint[1], 
                    'y': int(i - grid / 2) + maxPoint[0], 'referenceImage': refImage}
                # splineStrokes, splineStrokeColor = makeSplineStroke(canvas, initStroke)
                splineStrokes, splineStrokeColor = makeSplineStroke(canvas, initStroke)
                strokes.append({'R': radius, 'p': splineStrokes, 'c': splineStrokeColor,
                    'referenceImage': refImage})

    # paint all stroke in S on the canvas, in random order
    order = np.random.permutation(len(strokes))
    print(len(strokes))
    # currently paint in circles
    for o in order:
        # canvas = paintToCanvas(canvas, strokes[o])
        # canvas = paintToCanvasWithSource(canvas, strokes[o], source)
        canvas = paintSplineStrokesToCanvas(canvas, strokes[o], source)

    return canvas