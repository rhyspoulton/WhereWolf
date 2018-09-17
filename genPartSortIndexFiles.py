#!/usr/bin/env python
#
#This is a python file to create the Particle Sorted Index files needed by WhereWolf
#
# It uses mpi4py to process the gadget files in parallel
#
# It can be run by the command:
#
# $ python genPartSortIndexFiles.py <Gadget folder input directory> <Output directory> <snapshot>

import numpy as np 
import h5py
import time
import sys
import os

if(len(sys.argv)<4):
	raise SystemExit("Incorrect number of arguments parsed.\n \tUsage: python genPartSortIndexFiles.py <Gadget folder input directory> <Output directory> <snapshot>\n")


try:
	snap = int(sys.argv[3])
except ValueError:
	raise SystemExit("Please parse a int for the snapshot number")

GadDir = sys.argv[1]
OutputDir = sys.argv[2]

print("Doing snap",snap)
start = time.time()

#Lets find what type of file is present
GadFilename = GadDir + "/snapshot_%03d.hdf5" %snap
if(os.path.isfile(GadFilename)):
	impi = False
else:
	GadFilename = GadDir + "/snapshot_%03d.0.hdf5" %snap
	impi=True
	if(os.path.isfile(GadFilename)== False):
		raise IOError("Could not find file",GadFilename)


#Extract all the header info from the first file
GadFile = h5py.File(GadFilename,"r")
TotNpart = np.uint64(GadFile["Header"].attrs["NumPart_Total"][1])
NumFiles = GadFile["Header"].attrs["NumFilesPerSnapshot"]

#Lets see if the dataset which is used if the large amount of particles exists
if("NumPart_Total_HighWord" in GadFile["Header"].attrs):

	#Lets see if it is populated
	if(np.sum(GadFile["Header"].attrs["NumPart_Total_HighWord"])>0):
	       #Find the base and the power for the number of particles
	       sel = np.uint64(np.where(GadFile["Header"].attrs["NumPart_Total_HighWord"]>0)[0][0])
	       base = np.uint64(GadFile["Header"].attrs["NumPart_Total_HighWord"][sel])
	       power = np.uint64(32 + sel)

	       #Add this number to the current TotNpart
	       TotNpart += np.power(base,power)

GadFile.close()

#Lets check the number or particles has been read correctly
if(TotNpart<=0):
       raise IOError("Cannot determine the number of particle from the header")

#State how much memory is needed
print("To load in and sort",TotNpart,"particle IDs, this task will need minimum of",(2 * TotNpart * 8 )/( 1000**3),"GB of memory")

# Setup the values
pid = np.zeros(TotNpart,dtype=np.uint64)

if(impi):

	#Store the offsets for the number of particles in each file
	GadPartOffsets = np.zeros(NumFiles+1,dtype=np.uint64)
	GadNumPartFile = np.zeros(NumFiles,dtype=np.uint64)
	ioffset = np.uint64(0)

	# Loop over all the gadget files extacting them into 1 array
	for i in range(NumFiles):

		GadFilename = GadDir +"/snapshot_%03d.%i.hdf5" %(snap,i)
		GadFile = h5py.File(GadFilename,"r")
		GadNumPartFile[i] =  np.sum(GadFile["Header"].attrs["NumPart_ThisFile"],dtype=np.uint64)
		ioffset+=GadNumPartFile[i]

		#Load the particle IDs and offsets into the arrays
		pid[GadPartOffsets[i]:ioffset] = np.asarray(GadFile["PartType1"]["ParticleIDs"])
		GadPartOffsets[i+1] = ioffset
		GadFile.close()
else:

	#If just a single file then this is much simpler
	GadFile = h5py.File(GadFilename,"r")
	pid[:] = np.asarray(GadFile["PartType1"]["ParticleIDs"])
	GadFile.close()

#Argsort the file to find where and which file they are located
pidSortedIndexes = np.argsort(pid)

print("Done argsorting the Particle ID array now on to outputting the data")

if(impi):

	#Output the data in the same splitting that is present in the gadget files
	for i in range(NumFiles):

		hdffile = h5py.File(OutputDir+"/snapshot_%03d.WWpidsortindex.%i.hdf" %(snap,i),"w")

		#Create an attribute for the size and the number of particles this file
		hdffile.attrs["partOffset"]=GadPartOffsets[i]
		hdffile.attrs["fileNumPart"]=GadNumPartFile[i]

		#Output the dataset with the sorted indexes
		hdffile.create_dataset("pidSortedIndexes",data=pidSortedIndexes[GadPartOffsets[i]:GadPartOffsets[i+1]], compression='gzip', compression_opts=7)
		
		hdffile.close()
else:

	hdffile = h5py.File(OutputDir+"/snapshot_%03d.WWpidsortindex.hdf" %snap,"w")

	#Output the dataset with the sorted indexes
	hdffile.create_dataset("pidSortedIndexes",data=pidSortedIndexes,compression="gzip",compression_opts=7)

	hdffile.close()

#Free the memory to be used again
del pid
del pidSortedIndexes

print("Done creating file for snapshot",snap,"in",time.time()-start)

