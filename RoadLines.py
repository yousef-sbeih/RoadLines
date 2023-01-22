
"""
    this class is for road lines detection,
    it have one main static method: getRoadLines(image)
    that takes image as input and returns left and right lines coordinates
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
# constants that we need in processing
RHO = 1
THETA = np.pi/180
THRESHOLD = 40
MIN_LINE_LENGTH = 20
MAX_LINE_GAP = 20
YELLOW_LOWER_RANGE = np.array([30, 50, 50])
YELLOW_UPPER_RANGE = np.array([60, 255, 255])


class RoadLines:

    """
    convert image into gray scale for white color 
    and hsv for yellow color and return 2 copies of image
    """
    @staticmethod
    def convertImagesAndAppyGBlur(image):
        grayCopy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurForGray = cv2.GaussianBlur(grayCopy, (5, 5), 0)
        hsvCopy = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blurForHsv = cv2.GaussianBlur(hsvCopy, (5, 5), 0)
        return blurForGray, blurForHsv
    """
    this method, takes the output images of the previous method
    then apply yellow and white mask 
    and return the final mask after merging them.
    """
    @staticmethod
    def getMask(grayImage, hsvImage):
        whiteMask = cv2.inRange(grayImage, 200, 255)
        yellowMask = cv2.inRange(
            hsvImage, YELLOW_LOWER_RANGE, YELLOW_UPPER_RANGE)
        mainMask = cv2.bitwise_or(whiteMask, yellowMask)
        return mainMask
    """
    The following method is to apply canny method to get 
    the edges from the mask. Its apply gaussian blur again.    
    """
    @staticmethod
    def getEdges(mask):
        cannyEdges = cv2.Canny(mask, 100, 150)
        edgesWithBlur = cv2.GaussianBlur(cannyEdges, (5, 5), -1)
        return edgesWithBlur
    """
    The following method is to get only the region of interest 
    depends on image size.
    """
    @staticmethod
    def regionOfinterest(image):
        height, width = image.shape
        triangle = np.array([
            [(350, height), (1600, height), (950, 550)]
        ])
        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, triangle, 255)
        mask = cv2.bitwise_and(image, mask)
        return mask
    """
    This method is to get line slope
    """
    @staticmethod
    def getSlope(x1, x2, y1, y2):
        return (y2-y1) / (x2-x1)
    """
    The following method, takes the canny edges image, 
    and returns the lines from input image after filter them.
    """
    @staticmethod
    def getFilterdLines(image):
        edgesLines = cv2.HoughLinesP(image, RHO, THETA, THRESHOLD, np.array([]),
                                     minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
        filterdLines = []
        if edgesLines is not None:
            for line in edgesLines:
                for x1, y1, x2, y2 in line:
                    dx, dy = x2 - x1, y2 - y1
                    angle = np.arctan2(dy, dx)
                    angle = np.rad2deg(angle)
                    if (angle >= 40 and angle <= 50) or (angle >= -50 and angle <= -40):
                        filterdLines.append(line)
        return filterdLines
    """
    This method is to filterd lines to get only left and right lines
    """
    @staticmethod
    def getClosestLines(lines):
        halfWidth = 1920 / 2
        minLine = float('inf')
        maxLine = float('-inf')
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 > halfWidth:
                if x1 < minLine:
                    minLine = x1
                    rightLine = line
            else:
                if x1 > maxLine:
                    maxLine = x1
                    leftLine = line
        return [leftLine, rightLine]

    """
    This is the main method of the class that you can call it
    It takes image as input, apply all of method and return lines.
    """
    @staticmethod
    def getRoadLines(image):
        grayImage, hsvImage = RoadLines.convertImagesAndAppyGBlur(image)
        maskedImage = RoadLines.getMask(grayImage, hsvImage)
        cannyEdges = RoadLines.getEdges(maskedImage)
        ROIImage = RoadLines.regionOfinterest(cannyEdges)
        plt.imshow(ROIImage)
        plt.show()
        lines = RoadLines.getFilterdLines(ROIImage)
        filterdLines = RoadLines.getClosestLines(lines)
        return filterdLines
