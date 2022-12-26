"""
"""
from Common.Common import *
import cv2
import glob
from sklearn import svm
import joblib
import sys



class Classifier:
    def __init__(self):
        self.type = config.get("Classifier", "modelType")
        self.features = config.get("Classifier", "feature")
        self.training = boolCast(config.get("Classifier", "training"))
        self.testing =  boolCast(config.get("Classifier", "testing"))
        self.evaluation =  boolCast(config.get("Classifier", "evaluation"))
        self.storage = config.get("Classifier", "storagePath")
        self.trainingDataPath = config.get("Classifier", "trainingDataPath")
        self.testingDataPath = config.get("Classifier", "testingDataPath")
        self.version = int(config.get("Classifier", "version"))
        self.featureStoragePath = config.get("Classifier", "featureStoragePath")
        self.storagePath = config.get("Classifier", "storagePath")
        #config.set("Classifier", "version", str(int(self.version) + 1))
        self.preprocessing = boolCast(config.get("Classifier", "preprocessing"))
        if self.features == "HOG":
            self.HOGBlockSize = tupleCast(config.get("HOG", "blockSize"))
            self.HOGBlockStride = tupleCast(config.get("HOG", "blockStride"))
            self.HOGCellSize = tupleCast(config.get("HOG", "cellSize"))
            self.HOGNBins = int(config.get("HOG", "nBins"))
            self.HOGDerivAperture = int(config.get("HOG", "derivAperture"))
            self.HOGWinSigma = float(config.get("HOG", "winSigma"))
            self.HOGHistogramNormType = int(config.get("HOG", "histogramNormType"))
            self.HOGL2HysThreshold = float(config.get("HOG", "L2HysThreshold"))
            self.HOGGammaCorrection = int(config.get("HOG", "gammaCorrection"))
            self.HOGNLevels = int(config.get("HOG", "nLevels"))
            self.HOGWinSizeX = int(config.get("HOG", "winSizeX"))

def classifySquares(board: ChessBoard)-> ChessBoard:
    """
    This function is responsible for classifying the squares.
    :param board: The board to classify the squares from.
    :return: The classified squares.
    """
    pass

def trainClassifier(classifier: Classifier):
    """
    This function is responsible for training the classifier.
    :param classifier: The classifier to train.
    :return: The trained classifier.
    """
    if classifier.type == "SVM":
        trainSVMClassifier(classifier)

def trainSVMClassifier(classifier: Classifier):
    """
    This function is responsible for training the SVM piece classifier.
    :return: The trained SVM piece classifier.
    """
    if classifier.features == "HOG":
        if classifier.preprocessing:
            prepareHOGStorage(classifier)
            HOGImagePreprocessing(classifier)
        if classifier.training:
            prepareClassifierStorage(classifier)
            trainSVMHOGClassifier(classifier)
            postTrainingLogging(classifier)
        if classifier.testing:
            testSVMHOGClassifier(classifier)

def prepareHOGStorage(classifier: Classifier):
    """
    This function is responsible for preparing the HOG folder storage.
    :param classifier: The classifier to prepare the HOG storage for.
    :return: The prepared HOG storage.
    """
    if not os.path.exists(classifier.featureStoragePath):
        os.makedirs(classifier.featureStoragePath)
    for ratio in aspect_ratios:
        if not os.path.exists(os.path.join(classifier.featureStoragePath, str(ratio))):
            os.makedirs(os.path.join(classifier.featureStoragePath, str(ratio)))
        for piece in pieces:
            if not os.path.exists(os.path.join(classifier.featureStoragePath, str(ratio), piece)):
                os.makedirs(os.path.join(classifier.featureStoragePath, str(ratio), piece))

def HOGImagePreprocessing(classifier: Classifier):
    for i,ratio in enumerate(aspect_ratios):
        winSize = ( classifier.HOGWinSizeX, int (classifier.HOGWinSizeX * ratio) )
        hog = cv2.HOGDescriptor(winSize,classifier.HOGBlockSize,classifier.HOGBlockStride,classifier.HOGCellSize,
        classifier.HOGNBins,classifier.HOGDerivAperture,classifier.HOGWinSigma,classifier.HOGHistogramNormType,
        classifier.HOGL2HysThreshold,classifier.HOGGammaCorrection,classifier.HOGNLevels)
        for j,piece_dir in enumerate(pieces):
            for k,file in enumerate(glob.glob(classifier.trainingDataPath + piece_dir + "/*.jpg")):
                img = cv2.imread(file)
                img = cv2.resize(img, (classifier.HOGWinSizeX,classifier.HOGWinSizeX * 2))
                subImg = img[int(classifier.HOGWinSizeX*2-classifier.HOGWinSizeX*ratio):,:]
                features = hog.compute(subImg)
                np.save(os.path.join(classifier.featureStoragePath, str(ratio), piece_dir, os.path.basename(file) + ".npy"), features)

def prepareClassifierStorage(classifier: Classifier):
    """
    This function is responsible for preparing the classifier folder storage.
    :param classifier: The classifier to prepare the storage for.
    :return: The prepared classifier storage.
    """
    if not os.path.exists(classifier.storagePath):
        os.makedirs(classifier.storagePath)

def loadDataHOG(classifier,piece):
    X = None
    Y = None

    ratio = piece_to_ratio[piece]
    for piece_dir in pieces:
        piece_class = 0
        if piece == piece_dir:
            piece_class = 1

        for filename in glob.glob(os.path.join(classifier.featureStoragePath, str(ratio), piece_dir, "*.npy")):
            data = np.load(filename)
            if X is None:
                X = np.array(data.transpose())
                Y = np.array([piece_class])
            else:
                X = np.vstack( (X, data.transpose()) )
                Y = np.hstack( (Y, [piece_class]) )
    return (X, Y)

def trainSVMHOGClassifier(classifier: Classifier):
    """
    This function is responsible for training the SVM HOG classifier.
    :return: The trained SVM HOG classifier.
    """
    for piece in pieces:
        X, Y = loadDataHOG(classifier,piece)
        clf = svm.SVC(class_weight=piece_weights[piece], probability=True)
        clf.fit(X, Y)
        joblib.dump(clf, os.path.join(classifier.storagePath, piece + ".pkl"))

def postTrainingLogging(classifier: Classifier):
    """
    This function is responsible for logging the classifier training.
    :param classifier: The classifier to log the training for.
    :return: The logged classifier training.
    """
    config.set("Classifier", "version", str(int(classifier.version) + 1))
    with open(classifier.configPath, "w") as configfile:
        config.write(configfile)
    # Create text file with classifier configuration
    with open(os.path.join(classifier.storagePath, "classifier_config.txt"), "w") as text_file:
        text_file.write(f"Classifier version: {classifier.version}\n")
        text_file.write(f"Classifier type: {classifier.type}\n")
        text_file.write(f"Classifier features: {classifier.features}\n")
        text_file.write(f"Classifier preprocessing: {classifier.preprocessing}\n")
        text_file.write(f"Classifier training: {classifier.training}\n")
        text_file.write(f"Classifier testing: {classifier.testing}\n")
        text_file.write(f"Classifier training data path: {classifier.trainingDataPath}\n")
        text_file.write(f"Classifier feature storage path: {classifier.featureStoragePath}\n")
        text_file.write(f"Classifier storage path: {classifier.storagePath}\n")
        if classifier.type == "SVM":
            if classifier.features == "HOG":
                text_file.write(f"Classifier HOG win size X: {classifier.HOGWinSizeX}\n")
                text_file.write(f"Classifier HOG block size: {classifier.HOGBlockSize}\n")
                text_file.write(f"Classifier HOG block stride: {classifier.HOGBlockStride}\n")
                text_file.write(f"Classifier HOG cell size: {classifier.HOGCellSize}\n")
                text_file.write(f"Classifier HOG n bins: {classifier.HOGNBins}\n")
                text_file.write(f"Classifier HOG deriv aperture: {classifier.HOGDerivAperture}\n")
                text_file.write(f"Classifier HOG win sigma: {classifier.HOGWinSigma}\n")
                text_file.write(f"Classifier HOG histogram norm type: {classifier.HOGHistogramNormType}\n")


def testSVMHOGClassifier(classifier: Classifier):
    """
    This function is responsible for testing the SVM classifier.
    :return: The tested SVM classifier.
    """
    trainingImgsDir = classifier.trainingDataPath
    testingImgsDir = classifier.testingDataPath
    # redirect prints to log file
    sys.stdout = open(os.path.join(classifier.storagePath, "classifier_log.txt"), "w")
    print("Testing classifier")
    for piece in pieces:
        clf = joblib.load(os.path.join(classifier.storagePath, piece + ".pkl"))
        ratio = piece_to_ratio[piece]
        winSize = (classifier.HOGWinSizeX, int(classifier.HOGWinSizeX*ratio))
        hog = cv2.HOGDescriptor(winSize, classifier.HOGBlockSize, classifier.HOGBlockStride, classifier.HOGCellSize,
                                classifier.HOGNBins, classifier.HOGDerivAperture, classifier.HOGWinSigma,
                                classifier.HOGHistogramNormType, classifier.HOGL2HysThreshold, classifier.HOGGammaCorrection,
                                classifier.HOGNLevels)

        print(piece)
            

        # Training set
        num_correct = float(0)
        num_images = 0
        num_true_pos = 0
        num_true_neg = 0
        for piece_dir in pieces:
            num_correct_in_piece = 0
            for filename in glob.glob(os.path.join(trainingImgsDir, piece_dir, "*.jpg")):
                num_images = num_images + 1
                image = cv2.imread(filename)
                image = cv2.resize(image, (64, 128))
                image = image[int(128-64*ratio):,:]
                features = hog.compute(image)
                prediction = clf.predict(features.transpose())
                if piece == piece_dir and prediction[0] == 1:
                    num_true_pos = num_true_pos + 1
                    num_correct = num_correct + 1
                    num_correct_in_piece = num_correct_in_piece + 1
                elif piece != piece_dir and prediction[0] == 0:
                    num_true_neg = num_true_neg + 1
                    num_correct = num_correct + 1
                    num_correct_in_piece = num_correct_in_piece + 1
                # print(str(prediction) + " - " + str(piece_classes[piece_dir]))
            print(num_correct_in_piece)
        if num_images > 0:
            print("true pos: " + str(num_true_pos)+ " true neg: " +str(num_true_neg))
            print("num correct "+str(num_correct)+ " num of images "+ str(num_images))
            print("Train accuracy: " + str(num_correct/num_images))

        # Test set
        num_correct = float(0)
        num_images = 0
        num_true_pos = 0
        num_true_neg = 0
        for piece_dir in pieces:
            num_correct_in_piece = 0
            for filename in glob.glob(os.path.join(testingImgsDir, piece_dir, "*.jpg")):
                num_images = num_images + 1
                image = cv2.imread(filename)
                image = cv2.resize(image, (64, 128))
                image = image[int(128-64*ratio):,:]
                features = hog.compute(image)
                prediction = clf.predict(features.transpose())
                if piece == piece_dir and prediction[0] == 1:
                    num_true_pos = num_true_pos + 1
                    num_correct = num_correct + 1
                    num_correct_in_piece = num_correct_in_piece + 1
                elif piece != piece_dir and prediction[0] == 0:
                    num_true_neg = num_true_neg + 1
                    num_correct = num_correct + 1
                    num_correct_in_piece = num_correct_in_piece + 1
                # print(str(prediction) + " - " + str(piece_classes[piece_dir]))
            print(num_correct_in_piece)
        if num_images > 0:
            print(str(num_true_pos), str(num_true_neg))
            print(str(num_correct), str(num_images))
            print("Test accuracy: " + str(num_correct/num_images))



    

