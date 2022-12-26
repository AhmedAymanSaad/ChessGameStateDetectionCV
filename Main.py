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
    #csd.trainClassifier()

    
    
def main():
    mode = config.get("RunMode", "mode")
    #switch case for mode
    switcher = {
        "RunCSDonImg": RunCSDonImg,
        "TrainClassifier": TrainClassifier
    }
    func = switcher.get(mode, Default)
    func()   

if __name__ == "__main__":
    main()



