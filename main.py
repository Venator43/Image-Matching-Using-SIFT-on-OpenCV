import cv2
import numpy
import os
import matplotlib.pyplot as plt
from tkinter import *
import numpy as np
from PIL import Image, ImageTk

class Application():
	def __init__(self, master, imgPath):
		#load image database
		self.imgPath = imgPath
		self.imgDataset = []
		for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self.imgPath)):
			for f in filenames:
				file_path = os.path.join(dirpath, f)
				img = cv2.imread(file_path)
				img = cv2.resize(img,(250,250))
				self.imgDataset.append(img)
		
		imgShow1 = np.hstack((self.imgDataset[0], self.imgDataset[1], self.imgDataset[2]))
		imgShow2= np.hstack((self.imgDataset[3], self.imgDataset[4], self.imgDataset[5]))
		imgShow3 = np.hstack((self.imgDataset[6], self.imgDataset[7], self.imgDataset[8]))
		self.imgShow = np.vstack((imgShow1, imgShow2, imgShow3))

		
		while(True):
			cv2.imshow('gray',self.imgShow)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		#initialisasi variable dan objek untuk image matching
		self.userImage = np.array([])
		self.keypoints = 0
		self.descriptors = 0
		self.extractor = cv2.SIFT_create()

		#initialisasi tkinter
		self.master = master
		self.mainFrame = LabelFrame(self.master, text = "Image Matching", padx=5, pady=5)
		self.mainFrame.pack(padx=10, pady= 10)
		self.mainFrame.rowconfigure(0, minsize=300)
		self.mainFrame.columnconfigure(0, minsize=300)
		self.mainFrame.columnconfigure(1, minsize= 50)
		self.mainFrame.columnconfigure(2, minsize= 300)
		
		self.button1 = Button(self.mainFrame, text="Open Image", command=self.openImage)
		self.button1.grid(row=1,column=0)

		self.button2 = Button(self.mainFrame, text="Match Image", command=self.compare)
		self.button2.grid(row=1,column=2)

	def openImage(self):
		self.master.filename = filedialog.askopenfilename(initialdir="Gambar Test", title="Select a File", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
		curImage = cv2.imread(self.master.filename)
		curImage = cv2.cvtColor(curImage, cv2.COLOR_BGR2RGB)
		curImage = cv2.resize(curImage, (250,250))
		self.userImage = cv2.cvtColor(curImage, cv2.COLOR_RGB2GRAY)
		self.keypoints, self.descriptors = self.extractor.detectAndCompute(self.userImage,None)

		curImage = ImageTk.PhotoImage(Image.fromarray(curImage))
		self.imageFrame = Label(self.mainFrame, image = curImage)
		self.imageFrame.photo = curImage
		self.imageFrame.grid(row=0, column=0)

	def compare(self):
		imgMatch = []
		for i in range(len(self.imgDataset)):
			img2 = cv2.cvtColor(self.imgDataset[i], cv2.COLOR_BGR2GRAY);
		
			keypoints_2, descriptors_2 = self.extractor.detectAndCompute(img2,None)
			
			bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
			matches = bf.match(self.descriptors,descriptors_2)
			imgMatch.append(len(sorted(matches, key = lambda x:x.distance)))

		img = cv2.cvtColor(self.imgDataset[imgMatch.index(max(imgMatch))], cv2.COLOR_BGR2RGB)
		curImage = ImageTk.PhotoImage(Image.fromarray(img))
		self.imageMatch = Label(self.mainFrame, image = curImage)
		self.imageMatch.photo = curImage
		self.imageMatch.grid(row=0, column=2)

		print(imgMatch.index(max(imgMatch)))


if __name__ == '__main__':
	root = Tk()
	app = Application(root, "Gambar")
	root.mainloop()
