from ChessStateDetection import ChessStateDetection
from BoardDetection.BoardDetection import detectBoard

from Common.Common import *

def Default():
    print("Please enter a valid option.")

def RunCSDonImg():
    print ("Running CSD on image")
    pass

def TrainClassifier():
    print ("Training Classifier")
    csd = ChessStateDetection(None)
    testing = boolCast(config.get("TrainClassifier", "testing"))
    if testing:
        print("csd.trainClassifierCSD()")
        csd.trainClassifierCSD()

def TestClassifierOnBoard():
    print ("Testing Classifier on board")
    if boolCast(config.get("TestClassifierOnBoard", "readBoardFromCSV")):
        csd = ChessStateDetection(None)
        csd.readBoardFromCSV(config.get("TestClassifierOnBoard", "boardCSVPath"),io.imread(config.get("TestClassifierOnBoard", "boardImagePath")))
        csd.board.showBoardImageMarked()
        csd.board.GenerateChessSquareImages()
        csd.classifySquares()
        csd.printASCIIModifiedChessState()

def TestBoard():
    print ("Testing Board")
    testImgsDir = config.get("Directory", "testImagesDir")
    #loop over test images and call func
    for filename in os.listdir(testImgsDir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print("Testing image: " + filename)
            img = io.imread(testImgsDir + "/" + filename)
            detectBoard(img)
            continue
        else:
            continue

def TestSelBoard():
    print ("Testing Selected Board")
    img = io.imread(config.get("TestSelBoard", "imgPath"))
    csd = ChessStateDetection(img)
    csd.board.showBoardImageMarked()
    csd.classifySquares()
    csd.printASCIIModifiedChessState()
    csd.saveImagesClassified()

def TestLines():
    print ("Testing Lines")
    img = io.imread(config.get("TestLines", "imgPath"))
    csd = ChessStateDetection(img,True)
    csd.board.showBoardImageMarked()

def TestAllBoards():
    print ("Testing All Boards")
    testImgsDir = config.get("Directory", "testImagesDir")
    #loop over test images and call func
    for filename in os.listdir(testImgsDir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print("Testing image: " + filename)
            img = io.imread(testImgsDir + "/" + filename)
            csd = ChessStateDetection(img)
            csd.board.showBoardImageMarked(filename)
            csd.classifySquares()
            ascii = csd.printASCIIModifiedChessState()
            #save ascii to text file
            with open("output/boardsTest/" + filename + ".txt", "w") as text_file:
                text_file.write(ascii)
            continue
        else:
            continue


    
def main():
    mode = config.get("RunMode", "mode")
    #switch case for mode
    switcher = {
        "RunCSDonImg": RunCSDonImg,
        "TrainClassifier": TrainClassifier,
        "TestClassifierOnBoard": TestClassifierOnBoard,
        "TestBoard": TestBoard,
        "TestSelBoard": TestSelBoard,
        "TestLines": TestLines,
        "TestAllBoards": TestAllBoards
    }
    func = switcher.get(mode, Default)
    func()   

if __name__ == "__main__":
    main()



