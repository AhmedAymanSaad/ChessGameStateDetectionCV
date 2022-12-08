"""
This file contains all the static definitions pretianing to the project
"""

squareColors = { 0: "white", 1: "black" }
pieceColors = { 0: "white", 1: "black" }
pieceTypes = { 0: "empty", 1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king" }
pieceTypesInv = { "empty": 0, "pawn": 1, "knight": 2, "bishop": 3, "rook": 4, "queen": 5, "king": 6 }
pieceTypesShort = { 0: "e", 1: "p", 2: "n", 3: "b", 4: "r", 5: "q", 6: "k" }

# board enum
boardrows = { "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7 }
boardcols = { "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7 }

# ratios enum
aspectRatios = { "square": 1, "pawn": 1, "knight": 1.25, "bishop": 1.5, "rook": 1.25, "queen": 1.75, "king": 2 }

