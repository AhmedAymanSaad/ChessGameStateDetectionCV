"""
This is the main module responsible for running the chess state detection.
"""

from Common.Common import *
from BoardDetection.BoardDetection import *
from ChessPieceClassification.ChessPieceClassification import *

import numpy as np

class ChessStateDetection:
    def __init__(self, image: np.ndarray):
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
        count = 0
        for i in range(8):
            for j in range(8):
                if self.board.squares[count].piece == "empty":
                    print(".", end=" ")
                else:
                    if self.board.squares[count].color == "white":
                        print(piece_to_Notation_White[self.board.squares[count].piece], end=" ")
                    else:
                        print(piece_to_Notation_Black[self.board.squares[count].piece], end=" ")
                count += 1
            print()


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