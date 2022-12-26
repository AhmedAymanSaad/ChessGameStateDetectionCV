"""
This is the main module responsible for running the chess state detection.
"""

from Common.Common import *
from BoardDetection.BoardDetection import *
from ChessPieceClassification.ChessPieceClassification import *

import numpy as np

class ChessStateDetection:
    def __init__(self, image: np.ndarray):
        if image is not None:
            self.board = ChessBoard(image)
        self.boardAnalysis = False
        self.initClassifier()

    def processBoard(self):
        """
        This function is responsible for detecting the board.
        :param image: The image to detect the board from.
        :return: The detected board.
        """
        self.board = detectBoard(self.board)
        return self.board

    def classifySquares(self):
        """
        This function is responsible for classifying the squares.
        :return: The classified squares.
        """
        self.board = classifySquares(self.board)
        return self.board

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
        """
        if self.boardAnalysis:
            print("ASCII chess state") #TODO: Implement this function.
        pass

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
        trainClassifier(self.classifier)
        return self.classifier