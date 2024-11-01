import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def plot_image(img, fig_size=(8,6), cmap=None):
    if cmap is None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.close("all")
    plt.figure(figsize=fig_size)
    plt.imshow(img, cmap=cmap)
    plt.show()
   
def adjust_gamma(img, gamma=0.8):
    """Correct the brightness of an image by using a non-linear transformation
    between the input values and the mapped output values

    Args:
        img (array): The raw input image
        gamma (float): Gamma values < 1 will shift the image towards the darker end of the spectrum
                        while gamma values > 1 will make the image appear lighter

    Returns:
        The image after gamma correction
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def find_four_corners(size, contour):
    height, width = size
    borders = [[0,0], [0,height], [width, 0], [width, height]]
    max_dis = math.sqrt(height**2 + width**2)
    corners = [None, None, None, None]
    distance = [max_dis, max_dis, max_dis, max_dis]
    for point in contour:
        x,y = point
        for index, border in enumerate(borders):
            # Using eclude distance
            #temp_dis = math.sqrt((x-border[0])**2 + (y-border[1])**2)
            # Using mahattan distance
            temp_dis = math.sqrt(abs(x-border[0]) + abs(y-border[1]))
            if distance[index] > temp_dis:
                distance[index] = temp_dis
                corners[index] = [x,y]
    return corners

def get_rect(cnts):
    cnts_ = cnts.reshape(-1, 2)
    num = len(cnts_)
    x_min, y_min = np.min(cnts_, axis=0)
    x_max, y_max = np.max(cnts_, axis=0)
    extreme = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ])
    dist = np.sqrt(np.sum(cnts_ ** 2, axis=1).reshape(num, 1) +
                    np.sum(extreme ** 2, axis=1) - 2 * cnts_.dot(extreme.T))
    centerx = int((x_min + x_max) / 2)
    centery = int((y_min + y_max) / 2)
    center = np.array([[centerx, centery]])
    dist2 = np.sqrt(np.sum(cnts_ ** 2, axis=1).reshape(num, 1) +
                    np.sum(center ** 2, axis=1) - 2 * cnts_.dot(center.T))
    diff_dist = dist2 - dist
    corner_id = np.argmax(diff_dist, axis=0)
    return cnts[corner_id]

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped