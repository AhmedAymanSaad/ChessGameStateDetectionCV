"""
This file contains all the static definitions pretianing to the project
"""

import numpy as np

# chess pieces enums
pieces = ["empty", "pawn", "knight", "bishop", "rook", "queen", "king"]

# ratios enums
aspect_ratios = [1, 1.25, 1.5, 1.75, 2]
piece_to_ratio = {"empty": 1, "pawn": 1, "knight": 1.25, "bishop": 1.5, "rook": 1.75, "queen": 2, "king": 2}
# piece_weights = {"empty": {0: 64, 1: 272},
# 				 "pawn": {0: 128, 1: 208},
# 				 "knight": {0: 32, 1: 304},
# 				 "bishop": {0: 32, 1: 304}, # 304
# 				 "rook": {0: 32, 1: 304}, # 305
# 				 "queen": {0: 32, 1: 304}, # 306
# 				 "king": {0: 16, 1: 320}} # 320
# bishop	70		0.963692946	0.036307054		1858
# empty	1310		0.320539419	0.679460581		618
# king	45		0.976659751	0.023340249		1883
# knight	80		0.958506224	0.041493776		1848
# pawn	284		0.852697095	0.147302905		1644
# queen	53		0.972510373	0.027489627		1875
# rook	86		0.955394191	0.044605809		1842

piece_weights = {"empty": {0: 1310, 1: 618},
				 "pawn": {0: 128, 1: 208},
				 "knight": {0: 80, 1: 1848},
				 "bishop": {0: 70, 1: 1858}, 
				 "rook": {0: 86, 1: 1842}, 
				 "queen": {0: 53, 1: 1875}, 
				 "king": {0: 45, 1: 1883}} 

ratio_to_num = {1: 0, 1.25: 1, 1.5: 2, 1.75: 3, 2: 4}
piece_classes = {"empty": 0,
				 "pawn": 1,
				 "knight": 2,
				 "bishop": 3,
				 "rook": 4,
				 "queen": 5,
				 "king": 6}
piece_to_ratio_num = {"empty": 0,
					  "pawn": 0,
					"knight": 1,
					"bishop": 2,
					"rook": 1,
					"queen": 3,
					"king": 4}
piece_to_Notation = {"empty": " ",
					 "pawn": "P",
					 "knight": "N",
					 "bishop": "B",
					 "rook": "R",
					 "queen": "Q",
					 "king": "K"}
piece_to_Notation_White = {"empty": " ",
					 "pawn": "P",
					 "knight": "N",
					 "bishop": "B",
					 "rook": "R",
					 "queen": "Q",
					 "king": "K"}
piece_to_Notation_Black = {"empty": " ",
					 "pawn": "p",
					 "knight": "n",
					 "bishop": "b",
					 "rook": "r",
					 "queen": "q",
					 "king": "k"}

piece_not_to_num = {".": 0,
					"P": 1,
					"N": 2,
					"B": 3,
					"R": 4,
					"Q": 5,
					"K": 6,
					"p": 1,
					"n": 2,
					"b": 3,
					"r": 4,
					"q": 5,
					"k": 6}
