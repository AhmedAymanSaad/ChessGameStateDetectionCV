from ChessStateDetection import ChessStateDetection

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
        csd.printASCIIChessState()
    
def main():
    mode = config.get("RunMode", "mode")
    #switch case for mode
    switcher = {
        "RunCSDonImg": RunCSDonImg,
        "TrainClassifier": TrainClassifier,
        "TestClassifierOnBoard": TestClassifierOnBoard
    }
    func = switcher.get(mode, Default)
    func()   

if __name__ == "__main__":
    main()



