#Create gui page for CSD
#takes image as input 
#runs CSD on image and plots out the board
#shows the board image with the squares marked
#prints out the ASCII chess state
#shows the board image with the squares classified

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import os
import sys
import numpy as np
import shutil
import cv2
from skimage import io
from skimage import color
from skimage import img_as_ubyte
from skimage import img_as_float
from skimage import img_as_int
from skimage import img_as_uint
from skimage import img_as_bool

from skimage import data
from skimage import filters

from ChessStateDetection import ChessStateDetection
# import imageTK
from PIL import ImageTk, Image


#Create the window for the GUI
#contains a button to select an image
#contains a button to run CSD on the image
#contains a place to display the image after CSD has been run
#contains a place to display the ASCII chess state after CSD has been run

class CSDGUI:
    def __init__(self, master):
        self.master = master
        master.title("Chess State Detection")
        master.geometry("800x800")
        master.configure(background='white')
        
        #create a button to select an image
        self.selectImage = tk.Button(master, text="Select Image", command=self.selectImage)
        self.selectImage.pack()
        
        #create a button to run CSD on the image
        self.runCSD = tk.Button(master, text="Run CSD", command=self.runCSD)
        self.runCSD.pack()
        
        #create a place to display the image after CSD has been run
        self.image = tk.Label(master)
        self.image.pack()
        
        #create a text place to display the ASCII chess state after CSD has been run
        self.ascii = tk.Label(master)
        self.ascii.pack()


        
    def selectImage(self):
        #select an image
        self.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        
    def runCSD(self):
        #run CSD on the image
        self.image = io.imread(self.filename)
        self.csd = ChessStateDetection(self.image)
        boardImage = self.csd.board.showBoardImageMarked()
        self.csd.classifySquares()
        asciiMat = self.csd.printASCIIModifiedChessState()
        
        #display the image after CSD has been run and resize to fit the window keeping the aspect ratio in a part of the window
        self.image = ImageTk.PhotoImage(Image.fromarray(boardImage).resize((800, 800), Image.ANTIALIAS))
        self.imageLabel = tk.Label(image=self.image)
        self.imageLabel.pack()

        
        #display the text representation of the chess state after CSD has been run from asciiMat
        self.ascii = tk.Label(text=asciiMat)
        self.ascii.pack()
        

root = tk.Tk()
my_gui = CSDGUI(root)
root.mainloop()



