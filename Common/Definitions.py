"""
This file contains all the static definitions pretianing to the project
"""

# chess pieces enums
pieces = ["empty", "pawn", "knight", "bishop", "rook", "queen", "king"]

# ratios enums
aspect_ratios = [1, 1.25, 1.5, 1.75, 2]
piece_to_ratio = {"empty": 1, "pawn": 1, "knight": 1.25, "bishop": 1.5, "rook": 1.75, "queen": 2, "king": 2}
piece_weights = {"empty": {0: 64, 1: 272},
				 "pawn": {0: 128, 1: 208},
				 "knight": {0: 32, 1: 304},
				 "bishop": {0: 32, 1: 304}, # 304
				 "rook": {0: 32, 1: 304}, # 305
				 "queen": {0: 32, 1: 304}, # 306
				 "king": {0: 16, 1: 320}} # 320
