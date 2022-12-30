"""
This is the main module responsible for running the chess state detection.
"""

from Common.Common import *
from BoardDetection.BoardDetection import *
from ChessPieceClassification.ChessPieceClassification import *

import numpy as np
import shutil

class ChessStateDetection:
    def __init__(self, image: np.ndarray, prints = False):
        self.board = ChessBoard(image)
        if image is not None:
            self.board = self.processBoard(prints)
        self.boardAnalysis = False
        self.initClassifier()

    def processBoard(self, prints = False):
        """
        This function is responsible for detecting the board.
        :param image: The image to detect the board from.
        :return: The detected board.
        """
        corners, intersections = detectBoard(self.board.image, prints)
        self.board.boardDetection(corners, intersections)
        return self.board

    def classifySquares(self):
        """
        This function is responsible for classifying the squares.
        :return: The classified squares.
        """
        self.classifier.classifySquares(self.board)

    def getChessState(self):
        """
        This function is responsible for detecting the chess state.
        :return: The chess state.
        """
        self.board = self.processBoard()
        self.board = self.classifySquares()
        self.boardAnalysis = True
        return self.board

    def printASCIIChessState(self):
        """
        This function is responsible for printing the ASCII chess state.
        :return: The ASCII chess state.
        EG:
        r n b q k b n r
        p p p p p p p p
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P P P P P P P P
        R N B Q K B N R
        """
        squares = self.board.squares
        count = 0
        for i in range(8):
            for j in range(8):
                if squares[count].piece == "empty":
                    print(".", end=" ")
                else:
                    if squares[count].color == "white":
                        print(piece_to_Notation_White[squares[count].piece], end=" ")
                    else:
                        print(piece_to_Notation_Black[squares[count].piece], end=" ")
                count += 1
            print()

    def printASCIIModifiedChessState(self):
        # print ascii transposed and mirrored
        squares = self.board.squares
        count = 0
        printMat = [["0" for x in range(8)] for y in range(8)]
        for i in range(8):
            for j in range(8):
                if squares[count].piece == "empty":
                    printMat[i][j] = "."
                else:
                    if squares[count].color == "white":
                        printMat[i][j] = piece_to_Notation_White[squares[count].piece]
                    else:
                        printMat[i][j] = piece_to_Notation_Black[squares[count].piece]
                count += 1

        #rotate printMat 90 degree to left 
        printMat = np.rot90(printMat, 3)
        #mirror printMat
        printMat = np.flip(printMat, 1)
        for i in range(8):
            for j in range(8):
                print(printMat[i][j], end=" ")
            print()
        return printMat

    def saveImagesClassified(self):
        """
        This function is responsible for saving the images classified.
        :return: The images classified.
        """
        # delete output folder
        if os.path.exists("output"):
            shutil.rmtree("output")
        os.mkdir("output")
        # save images
        countEmpty =0
        for square in self.board.squares:
            if square.piece != "empty":
                cv2.imwrite("output/" + square.piece + "_" + square.color + ".png", square.image[4])
            else:
                cv2.imwrite("output/" + square.piece + "_" + str(countEmpty) + ".png", square.image[4])
                countEmpty += 1




    def initClassifier(self):
        """
        This function is responsible for initializing the classifier.
        :return: The initialized classifier.
        """
        classifier = Classifier()
        self.classifier = classifier

    def trainClassifierCSD(self):
        """
        This function is responsible for training the classifier.
        :return: The trained classifier.
        """
        if not self.classifier.training:
            self.classifier.version = int(config.get("Classifier", "currVersion"))
        self.classifier.trainClassifier()
        return self.classifier

    def readBoardFromCSV(self, boardCSVPath: str, boardImagePath: str):
        """
        This function is responsible for reading the board from a CSV file.
        :return: The board read from the CSV file.
        """
        self.board.readBoardFromCSV(boardCSVPath, boardImagePath)
        return self.board