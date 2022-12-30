"""
"""
from Common.Common import *
import cv2
import glob
from sklearn import svm
import joblib
import sys
import timeit
import multiprocessing




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
        self.mulitprocessing = boolCast(config.get("Common", "multiprocessing"))
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

    def classifySquares(self, board):
        """
        This function is responsible for classifying the square.
        :param square: The square to classify.
        :return: The classified square.
        """
        if self.type== "SVM" and self.features == "HOG":
            start = timeit.default_timer()
            if self.mulitprocessing:
                #use parallel processing to classify the squares
                pool = multiprocessing.Pool()
                pool = multiprocessing.Pool(processes=8)
                squares = pool.map(self.classifySquareHOG, board.squares)
                board.squares = squares
            else:
                for square in board.squares:
                    square = self.classifySquareHOG(square)
            stop = timeit.default_timer()
            print("Classification time: ", stop - start)
            board.boardAnalysis = True

    def classifySquareHOG(self, square):
        """
        This function is responsible for classifying the square using HOG features.
        :param square: The square to classify.
        :return: The classified square.
        """
        predictions = np.zeros((len(pieces), 2))
        probabilities = []
        for piece in pieces:
            img = square.image[ratio_to_num[2.0]]
            winSize = (self.HOGWinSizeX, int(self.HOGWinSizeX*piece_to_ratio[piece]))
            hog = cv2.HOGDescriptor(winSize, self.HOGBlockSize, self.HOGBlockStride, self.HOGCellSize, self.HOGNBins,
                                    self.HOGDerivAperture, self.HOGWinSigma, self.HOGHistogramNormType,
                                    self.HOGL2HysThreshold, self.HOGGammaCorrection, self.HOGNLevels)
            img = cv2.resize(img, (self.HOGWinSizeX, self.HOGWinSizeX*2))
            img = img[int(self.HOGWinSizeX*2-self.HOGWinSizeX*piece_to_ratio[piece]):,:]
            features = hog.compute(img)
            clf = joblib.load(os.path.join(self.storagePath, piece + ".pkl"))
            prediction = clf.predict_proba(features.transpose())
            predictions[piece_classes[piece]] = prediction[0]
            probabilities.append(prediction[0][1])
        square.piece = pieces[np.argmax(probabilities)]
        square.probability = np.max(probabilities)
        if square.piece != "empty":
            # check the color of the piece
            img = square.image[ratio_to_num[1.0]]
            img = cv2.resize(img, (self.HOGWinSizeX, self.HOGWinSizeX*2))
            img = img[int(128-64*piece_to_ratio[piece]):,:]
            # check the color of the middle 8 pixels of the piece
            midX = int(img.shape[1]/2)
            midY = int(img.shape[0]/2)
            midImg = img[midY-4:midY+4, midX-4:midX+4]
            # turn the image to grayscale
            midImg = cv2.cvtColor(midImg, cv2.COLOR_BGR2GRAY)
            # check if color is black or white
            if np.mean(midImg) > 100:
                square.color = "white"
            else:
                square.color = "black"

        return square
            
            

    def trainClassifier(self):
        """
        This function is responsible for training the classifier.
        :param classifier: The classifier to train.
        :return: The trained classifier.
        """
        if self.type == "SVM":
            self.trainSVMClassifier(self)

    def trainSVMClassifier(self):
        """
        This function is responsible for training the SVM piece classifier.
        :return: The trained SVM piece classifier.
        """
        if self.features == "HOG":
            if self.preprocessing:
                self.prepareHOGStorage(self)
                self.HOGImagePreprocessing(self)
            if self.training:
                self.prepareClassifierStorage(self)
                self.trainSVMHOGClassifier(self)
                self.postTrainingLogging(self)
            if self.testing:
                self.testSVMHOGClassifier(self)

    def prepareHOGStorage(self):
        """
        This function is responsible for preparing the HOG folder storage.
        :param classifier: The classifier to prepare the HOG storage for.
        :return: The prepared HOG storage.
        """
        if not os.path.exists(self.featureStoragePath):
            os.makedirs(self.featureStoragePath)
        for ratio in aspect_ratios:
            if not os.path.exists(os.path.join(self.featureStoragePath, str(ratio))):
                os.makedirs(os.path.join(self.featureStoragePath, str(ratio)))
            for piece in pieces:
                if not os.path.exists(os.path.join(self.featureStoragePath, str(ratio), piece)):
                    os.makedirs(os.path.join(self.featureStoragePath, str(ratio), piece))

    def HOGImagePreprocessing(self):
        for i,ratio in enumerate(aspect_ratios):
            winSize = ( self.HOGWinSizeX, int (self.HOGWinSizeX * ratio) )
            hog = cv2.HOGDescriptor(winSize,self.HOGBlockSize,self.HOGBlockStride,self.HOGCellSize,
            self.HOGNBins,self.HOGDerivAperture,self.HOGWinSigma,self.HOGHistogramNormType,
            self.HOGL2HysThreshold,self.HOGGammaCorrection,self.HOGNLevels)
            for j,piece_dir in enumerate(pieces):
                for k,file in enumerate(glob.glob(self.trainingDataPath + piece_dir + "/*.jpg")):
                    img = cv2.imread(file)
                    img = cv2.resize(img, (self.HOGWinSizeX,self.HOGWinSizeX * 2))
                    subImg = img[int(self.HOGWinSizeX*2-self.HOGWinSizeX*ratio):,:]
                    features = hog.compute(subImg)
                    np.save(os.path.join(self.featureStoragePath, str(ratio), piece_dir, os.path.basename(file) + ".npy"), features)

    def prepareClassifierStorage(self):
        """
        This function is responsible for preparing the classifier folder storage.
        :param classifier: The classifier to prepare the storage for.
        :return: The prepared classifier storage.
        """
        if not os.path.exists(self.storagePath):
            os.makedirs(self.storagePath)

    def loadDataHOG(self,piece):
        X = None
        Y = None

        ratio = piece_to_ratio[piece]
        for piece_dir in pieces:
            piece_class = 0
            if piece == piece_dir:
                piece_class = 1

            for filename in glob.glob(os.path.join(self.featureStoragePath, str(ratio), piece_dir, "*.npy")):
                data = np.load(filename)
                if X is None:
                    X = np.array(data.transpose())
                    Y = np.array([piece_class])
                else:
                    X = np.vstack( (X, data.transpose()) )
                    Y = np.hstack( (Y, [piece_class]) )
        return (X, Y)

    def trainSVMHOGClassifier(self):
        """
        This function is responsible for training the SVM HOG classifier.
        :return: The trained SVM HOG classifier.
        """
        for piece in pieces:
            X, Y = self.loadDataHOG(piece)
            clf = svm.SVC(class_weight=piece_weights[piece], probability=True)
            clf.fit(X, Y)
            joblib.dump(clf, os.path.join(self.storagePath, piece + ".pkl"))

    def postTrainingLogging(self):
        """
        This function is responsible for logging the classifier training.
        :param classifier: The classifier to log the training for.
        :return: The logged classifier training.
        """
        config.set("Classifier", "version", str(int(self.version) + 1))
        with open(self.configPath, "w") as configfile:
            config.write(configfile)
        # Create text file with classifier configuration
        with open(os.path.join(self.storagePath, "classifier_config.txt"), "w") as text_file:
            text_file.write(f"Classifier version: {self.version}\n")
            text_file.write(f"Classifier type: {self.type}\n")
            text_file.write(f"Classifier features: {self.features}\n")
            text_file.write(f"Classifier preprocessing: {self.preprocessing}\n")
            text_file.write(f"Classifier training: {self.training}\n")
            text_file.write(f"Classifier testing: {self.testing}\n")
            text_file.write(f"Classifier training data path: {self.trainingDataPath}\n")
            text_file.write(f"Classifier feature storage path: {self.featureStoragePath}\n")
            text_file.write(f"Classifier storage path: {self.storagePath}\n")
            if self.type == "SVM":
                if self.features == "HOG":
                    text_file.write(f"Classifier HOG win size X: {self.HOGWinSizeX}\n")
                    text_file.write(f"Classifier HOG block size: {self.HOGBlockSize}\n")
                    text_file.write(f"Classifier HOG block stride: {self.HOGBlockStride}\n")
                    text_file.write(f"Classifier HOG cell size: {self.HOGCellSize}\n")
                    text_file.write(f"Classifier HOG n bins: {self.HOGNBins}\n")
                    text_file.write(f"Classifier HOG deriv aperture: {self.HOGDerivAperture}\n")
                    text_file.write(f"Classifier HOG win sigma: {self.HOGWinSigma}\n")
                    text_file.write(f"Classifier HOG histogram norm type: {self.HOGHistogramNormType}\n")


    def testSVMHOGClassifier(self):
        """
        This function is responsible for testing the SVM classifier.
        :return: The tested SVM classifier.
        """
        trainingImgsDir = self.trainingDataPath
        testingImgsDir = self.testingDataPath
        # redirect prints to log file
        sys.stdout = open(os.path.join(self.storagePath, "classifier_log.txt"), "w")
        print("Testing classifier")
        for piece in pieces:
            clf = joblib.load(os.path.join(self.storagePath, piece + ".pkl"))
            ratio = piece_to_ratio[piece]
            winSize = (self.HOGWinSizeX, int(self.HOGWinSizeX*ratio))
            hog = cv2.HOGDescriptor(winSize, self.HOGBlockSize, self.HOGBlockStride, self.HOGCellSize,
                                    self.HOGNBins, self.HOGDerivAperture, self.HOGWinSigma,
                                    self.HOGHistogramNormType, self.HOGL2HysThreshold, self.HOGGammaCorrection,
                                    self.HOGNLevels)

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

        



    

