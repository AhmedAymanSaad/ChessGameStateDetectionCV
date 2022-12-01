"""
This is the main module responsible for running the chess state detection.
"""

from Common import *
import BoardDetection
import ChessPieceClassification
import ChessBoard

class ChessStateDetection:
    def __init__(self, image: np.ndarray):
        self.board = ChessBoard(image)
        self.boardAnalysis = False

    def processBoard(self):
        """
        This function is responsible for detecting the board.
        :param image: The image to detect the board from.
        :return: The detected board.
        """
        self.board = BoardDetection.detectBoard(self.board)
        return self.board

    def classifySquares(self):
        """
        This function is responsible for classifying the squares.
        :return: The classified squares.
        """
        self.board = ChessPieceClassification.classifySquares(self.board)
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