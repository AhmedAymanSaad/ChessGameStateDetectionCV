from configparser import ConfigParser

import ChessStateDetection as csd

def Default():
    print("Please enter a valid option.")

def RunCSDonImg():
    pass

def TestBoardDetection():
    pass

def ModelTraining():
    pass
    
def main():
    config = ConfigParser()
    config.read("Common\Config.ini")
    mode = config.get("RunMode", "mode")
    #switch case for mode
    switcher = {
        "RunCSDonImg": RunCSDonImg,
        "TestBoardDetection": TestBoardDetection,
        "ModelTraining": ModelTraining
    }
    func = switcher.get(mode, Default)
    func()   

if __name__ == "__main__":
    main()



