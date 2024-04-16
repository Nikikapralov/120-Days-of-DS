from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy import ndarray

import cv2
import numpy as np

"""
https://www.youtube.com/watch?v=SQ3D1tlCtNg

How to do perspective correction?
1. Get a photo of the object with a background that makes it stand out and is uniform. White paper for example.
2. Turn the photo to grayscale.
3. Blur the details in the photo as much as possible, but make the edges stand out.
4. Find all the contours in the photo using an algorithm (Canny for example).
5. Find the biggest contour.
6. Approximate the contour to a shape with 4 points (if looking for a card - rectangle).
7. Calculate the new points from the old points.
8. Use the cv2 provided functions to do perspective correction.
"""

def biggest_contour(contours: list[ndarray]) -> ndarray:
    """
    Find the biggest contour by its max array.
    For each contour in the contours.
    Find the area.
    Filter out small contours.
    Find a shape that is closed.
    Approximate the shape and make sure the curve is cloded.
    Make sure the contour has 4 corners.
    :param contours: A list of all contours that we have found.
    :return: The positions of the biggest contour.
    """
    biggest: ndarray = np.array([])
    max_area: float = 0
    for i in contours:
        area: float = cv2.contourArea(i)
        if area > 1000:  # Filter out small contours
            peri: float = cv2.arcLength(i, True)  # Makes sure the shape is closed, has 4 corners
            approx: ndarray = cv2.approxPolyDP(i, 0.015 * peri, True)  # Approx to another shale and curve is closed.
            if area > max_area and len(approx) == 4:  # Make sure the contour has 4 corners.
                biggest: ndarray = approx
                max_area: float = area
    return biggest


img: ndarray = cv2.imread('document.jpg') # Read the image.
img: ndarray = cv2.resize(img, (250, 250))
img_original: ndarray = img.copy() # Copy the image

# Image modification
gray: ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts and image to a different colorspace
gray: ndarray = cv2.bilateralFilter(gray, 100, 10, 50)

"""
The bilateralFilter function in OpenCV is used for smoothing images while preserving edges.
It's a non-linear filter that replaces the intensity of each pixel with a weighted average
of its neighboring pixels, based on both spatial proximity and intensity similarity.
This means that nearby pixels that are similar in intensity contribute more to the
smoothing process, while pixels with distinct intensity values are preserved.
"""

edged: ndarray = cv2.Canny(gray, threshold1=10, threshold2=15) # Between and above are edges. (between only if connected to edges)

"""
The Canny edge detector is an edge detection operator that uses
a multi-stage algorithm to detect a wide range of edges in images. 
"""

contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

"""
The findContours function in OpenCV (cv2) is used to detect contours in a binary image.
Contours are simply outlines or curves representing the boundaries of objects in an image.
This function helps in various image processing tasks such as object detection,
shape analysis, and segmentation.
"""
contours: list[ndarray] = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Sort them by biggest area

biggest: ndarray = biggest_contour(contours)  # Find biggest contour

cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)  # Draw the contours on the image.

# Pixel values in the original image
points: ndarray = biggest.reshape(4, 2)  # Reshapes the array so we have 4 corners with 2 points
input_points: ndarray = np.zeros((4, 2), dtype="float32")  # Mark the input points

# Get all points
points_sum: ndarray = points.sum(axis=1)
input_points[0]: ndarray = points[np.argmin(points_sum)]
input_points[3]: ndarray = points[np.argmax(points_sum)]

points_diff: ndarray = np.diff(points, axis=1)
input_points[1]: ndarray = points[np.argmin(points_diff)]
input_points[2]: ndarray = points[np.argmax(points_diff)]

(top_left, top_right, bottom_right, bottom_left) = input_points

# Get distances between points
bottom_width: float = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
top_width: float = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
right_height: float = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
left_height: float = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

# Output image size
max_width: int = max(int(bottom_width), int(top_width))
# max_height = max(int(right_height), int(left_height))
max_height: int = int(max_width * 1.414)  # for A4

# Desired points values in the output image
converted_points: ndarray = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

# Perspective transformation
matrix: ndarray = cv2.getPerspectiveTransform(input_points, converted_points)
img_output: ndarray = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

# Image shape modification for hstack
gray: ndarray = np.stack((gray,) * 3, axis=-1)
edged: ndarray = np.stack((edged,) * 3, axis=-1)

img_hor: ndarray = np.hstack((img_original, gray, edged, img))
cv2.imshow("Contour detection", img_hor)
cv2.imshow("Warped perspective", img_output)

# cv2.imwrite('output/document.jpg', img_output)

cv2.waitKey(0)