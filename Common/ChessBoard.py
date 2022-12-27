"""

"""
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import draw

from Common.Definitions import *


class ChessSquare:
    def __init__(self):
        self.image = [] # aspect ratio
        self.piece = None
        self.color = None
        self.corners = [] 
        self.center = None
        self.rank = None #row
        self.file = None #col
        


class ChessBoard:
    def __init__(self, image: np.ndarray):
        """
        This is the constructor of the class.
        :param image: The image to detect the board from.
        """
        self.image = image
        self.corners = []
        self.squares = []
    
    def readBoardFromCSV(self,path : str = None, image : np.ndarray = None):
        """
        This function is responsible for reading the board from a CSV file.
        :return: The board read from CSV.
        """
        self.image = image
        self.read_corners( path)
        self.read_intersections(path)
        self.GenerateChessSquareImages()

    #read corners from csv file
    def read_corners(self, path : str = None):
        with open(path + '/corners.csv', 'r') as f:
            for line in f:
                xy = line.split(',')
                self.corners.append([int(float(xy[1])), int(float(xy[0]))])
                    
        return self

    #read intersections from csv file into numpy 2d array first line is x and second line is y
    def read_intersections(self , path : str = None):
        intersections = np.zeros((81, 2))
        with open(path + '/intersections.csv', 'r') as f:
            for i, line in enumerate(f):
                for j, x in enumerate(line.split(',')):
                    intersections[j][(i+1)%2] = int(float(x))
        for i in range(64):
            squareCorners = np.zeros((4, 2))
            row = i%8
            col = math.floor(i/8)
            squareCorners[0] = intersections[col*9 + row].astype(int)
            squareCorners[1] = intersections[col*9 + row + 1].astype(int)
            squareCorners[2] = intersections[(col+1)*9 + row + 1].astype(int)
            squareCorners[3] = intersections[(col+1)*9 + row].astype(int)
            squareCenter = np.mean(squareCorners, axis=0)
            chessSquare = ChessSquare()
            chessSquare.center = squareCenter
            chessSquare.corners = squareCorners.astype(int)
            self.squares.append(chessSquare)
        return self

    def showBoardImageMarked(self):
        """
        This function is responsible for showing the board image marked with corners and intersections.
        :return: The board image marked.
        """
        imageProcessed = self.image.copy()
        for square in self.squares:
            r, c = draw.disk([square.center[0], square.center[1]], 30)
            imageProcessed[r, c, :] = [255, 0, 0]

            for corner in square.corners:
                r, c = draw.disk([corner[0], corner[1]], 30)
                imageProcessed[r, c, :] = [0, 0, 255]

        for corner in self.corners:
            r, c = draw.disk([corner[0], corner[1]], 60)
            imageProcessed[r, c, :] = [0, 255, 0]

        plt.imshow(imageProcessed)
        plt.show()

    def GenerateChessSquareImages(self):
        for square in self.squares:
            square.image = []
            for aspect_ratio in aspect_ratios:
                c1 = square.corners[0]
                c2 = square.corners[1]
                c3 = square.corners[2]
                c4 = square.corners[3]
                maxX = max(c1[0], c2[0], c3[0], c4[0])
                maxY = max(c1[1], c2[1], c3[1], c4[1])
                minX = min(c1[0], c2[0], c3[0], c4[0])
                minY = min(c1[1], c2[1], c3[1], c4[1])
                diff = (maxX - minX)
                # imageRs = cv2.resize(chessBoard.image, (64, 128))
                # subImg = imageRs[minX:maxX, minY:maxY]
                # subImgRs = cv2.resize(subImg, (int(64*aspect_ratio), 64))
                sqImg = self.image[minX-int(diff*aspect_ratio):maxX, minY:maxY]
                # sqImgRs = cv2.resize(sqImg, (int(64*aspect_ratio), 64))
                square.image.append(sqImg)

    def DisplayChessSquareImages(self):
        for j,square in enumerate(self.squares):
            for i, image in enumerate(square.image):
                if j == 56 or j == 0:
                    plt.imshow(image)
                    plt.show()

            plt.show() 
    
