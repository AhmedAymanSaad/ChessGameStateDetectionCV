"""

"""
import numpy as np

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
