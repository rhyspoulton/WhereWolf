import numpy as np
import os

class WWOptions(object):

	def __init__(self,tmpOpt):

		GadFileListFname = None
		VELFileListFname = None
		TreeFileListFname = None
		WWPIDSortedIndexListFname = None

		#Set the defaults for the parameters
		self.outputdir = tmpOpt.outputdir
		self.numsnaps = tmpOpt.numsnaps
		self.GadFileList = []
		self.VELFileList = []
		self.TreeFileList = []
		self.WWPIDSortedIndexList = []

		self.Num_Halos_search = 100
		self.iCore = 1
		self.CoreFrac = 0.4
		self.MinNumpartCore = 20
		self.NumSnapsWithinCoreMerge = 3

		self.Temporal_haloidval = 1000000000000
		self.Snapshot_offset = 0

		#Lets check if the file exists
		# if(not os.path.isfile(filename)):
		# 	raise IOError("No config file parsed")

		with tmpOpt.configfile as f:

			for line in f:

				if(line[0]=="#"): continue

				line = line.replace(" ","")

				line = line.strip()

				if(not line): continue

				line = line.split("=")

				if(line[0]=="GadFileList"):
					GadFileListFname = line[1]

				elif(line[0]=="VELFileList"):
					VELFileListFname = line[1]

				elif(line[0]=="TreeFileList"):
					TreeFileListFname = line[1]

				elif(line[0]=="WWPIDSortedIndexList"):
					WWPIDSortedIndexListFname = line[1]

				elif(line[0]=="Num_Halos_search"):
					self.Num_Halos_search = int(line[1])

				elif(line[0]=="iCore"):
					self.iCore = int(line[1])

				elif(line[0]=="CoreFrac"):
					self.CoreFrac = float(line[1])

				elif(line[0]=="MinNumpartCore"):
					self.MinNumpartCore = int(line[1])

				elif(line[0]=="NumSnapsWithinCoreMerge"):
					self.NumSnapsWithinCoreMerge = int(line[1])

				elif(line[0]=="Temporal_haloidval"):
					self.Temporal_haloidval = np.int64(line[1])

				elif(line[0]=="Snapshot_offset"):
					self.Snapshot_offset = int(line[1])

				else:
					raise OSError("Invalid config option %s, please only use the options in the sample config file" %line[0])

		tmpOpt.configfile.close()

		#Now lets check the config options
		if(GadFileListFname==None):
			raise IOError("No file that constains a list of the gadget base filename for each snapshot, please check if this is set in the config file")
		elif(VELFileListFname==None):
			raise IOError("No file that constains a list of the VELOCIraptor base filename for each snapshot, please check if this is set in the config file")
		elif(TreeFileListFname==None):
			raise IOError("No file that constains a list of the TreeFrog base filename for each snapshot, please check if this is set in the config file")
		elif(WWPIDSortedIndexListFname==None):
			raise IOError("No file that constains a list of the Particle ID's sorted index base filenames for each snapshot, please check if this is set in the config file")

		#Open up the files containing the list of snapshots
		GadFileListFile = open(GadFileListFname,"r")
		VELFileListFile = open(VELFileListFname,"r")
		TreeFileListFile = open(TreeFileListFname,"r")
		WWPIDSortedIndexListFile = open(WWPIDSortedIndexListFname,"r")

		# #Lets extract all the filenames, making sure there is numsnaps in each file
		self.GadFileList = [line.rstrip() for line in GadFileListFile]
		self.VELFileList = [line.rstrip() for line in VELFileListFile]
		self.TreeFileList = [line.rstrip() for line in TreeFileListFile]
		self.WWPIDSortedIndexList = [line.rstrip() for line in WWPIDSortedIndexListFile]

		GadFileListFile.close()
		VELFileListFile.close()
		TreeFileListFile.close()
		WWPIDSortedIndexListFile.close()


		# Lets check if the lists have length of greater or equal to the opt.numsnaps
		if(len(self.GadFileList)<self.numsnaps):
			raise IOError("The file %s only contains %i snapshots, but %i snapshots was requested. \nPlease add more files to this list or adjust the number of snapshots to %i" %(GadFileListFname,len(self.GadFileList),self.numsnaps,len(self.GadFileList)))
		elif(len(self.VELFileList)<self.numsnaps):
			raise IOError("The file %s only contains %i snapshots, but %i snapshots was requested. \nPlease add more files to this list or adjust the number of snapshots to %i" %(VELFileListFname,len(self.VELFileList),self.numsnaps,len(self.VELFileList)))
		elif(len(self.TreeFileList)<self.numsnaps):
			raise IOError("The file %s only contains %i snapshots, but %i snapshots was requested. \nPlease add more files to this list or adjust the number of snapshots to %i" %(TreeFileListFname,len(self.TreeFileList),self.numsnaps,len(self.TreeFileList)))
		elif(len(self.WWPIDSortedIndexList)<self.numsnaps):
			raise IOError("The file %s only contains %i snapshots, but %i snapshots was requested. \nPlease add more files to this list or adjust the number of snapshots to %i" %(WWPIDSortedIndexListFname,len(self.WWPIDSortedIndexList),self.numsnaps,len(self.WWPIDSortedIndexList)))

