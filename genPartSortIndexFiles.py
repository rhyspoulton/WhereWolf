#!/usr/bin/env python
#
#This is a python file to create the Particle Sorted Index files needed by WhereWolf
#
# It uses mpi4py to process the gadget files in parallel
#
# It can be run by the command:
#
# $ mpirun -n <num threads> python genPartSortIndexFiles.py <Gadget folder input directory> <Output directory> <initial snapshot> <final snapshot>

import numpy as np 
import h5py
from mpi4py import MPI
import time
import sys

comm = MPI.COMM_WORLD
Rank = comm.Get_rank()
size = comm.Get_size()


if(len(sys.argv)<5):
	raise SystemExit("Incorrect number of arguments parsed.\n \tUsage: mpirun -n <num threads> python genPartSortIndexFiles.py <Gadget folder input directory> <Output directory> <initial snapshot> <final snapshot>\n")


try:
	isnap = int(sys.argv[3])
	fsnap = int(sys.argv[4])
except ValueError:
	raise SystemExit("Please parse a int for the snapshot number")

GadDir = sys.argv[1]
OutputDir = sys.argv[2]

numsnaps = fsnap - isnap + 1

#Let the first process decide which snapshots each process should do
if(Rank==0):

	#Find the amount of snapshots per process
	numsnapsPerProcess = int(np.floor(numsnaps/size))

	snapRanges = []

	processIsnap=isnap
	processFsnap=numsnapsPerProcess + isnap

	for i in range(size):

		#If the final process, then make sure it includes the final snapshot
		if(i==size-1):
			snapRanges.append(range(processIsnap,fsnap))
		else:
			snapRanges.append(range(processIsnap,processFsnap))

		processIsnap+=numsnapsPerProcess
		processFsnap+=numsnapsPerProcess

else:
	#If not the root process then just define a varible to be assigned in a broadcast
	snapRanges=None

#Broadcast the snapranges to each process
snapRanges = comm.bcast(snapRanges,root=0)

print(Rank,"is doing snaps",list(snapRanges[Rank])[0],"to",list(snapRanges[Rank])[-1])
for snap in snapRanges[Rank]:

	print(Rank,"doing snap",snap)
	start = time.time()

	#Extract all the header info from the first file
	GadFilename = GadDir + "/snapshot_%03d.0.hdf5" %snap
	GadFile = h5py.File(GadFilename,"r")
	TotNpart = np.sum(GadFile["Header"].attrs["NumPart_Total"])
	NumFiles = GadFile["Header"].attrs["NumFilesPerSnapshot"]

	# Setup the values
	pid = np.zeros(TotNpart,dtype=np.uint64)

	#Store the offsets for the number of particles in each file
	GadFileOffsets = np.zeros(NumFiles+1,dtype=np.uint64)
	ioffset = np.uint64(0)

	# Loop over all the gadget files extacting them into 1 array
	for i in range(NumFiles):

		GadFilename = GadDir +"/snapshot_%03d.%i.hdf5" %(snap,i)
		GadFile = h5py.File(GadFilename,"r")
		numPartThisFile =  np.sum(GadFile["Header"].attrs["NumPart_ThisFile"],dtype=np.uint64)
		ioffset+=numPartThisFile

		#Load the particle IDs and offsets into the arrays
		pid[GadFileOffsets[i]:ioffset] = np.asarray(GadFile["PartType1"]["ParticleIDs"])
		GadFileOffsets[i+1] = ioffset
		GadFile.close()

	#Argsort the file to find where and which file they are located
	pidSortedIndexes = np.argsort(pid)

	print(Rank,"done argsorting the Particle ID array now on to outputting the data")

	#Output the data in the same splitting that is present in the gadget files
	for i in range(NumFiles):

		hdffile = h5py.File(OutputDir+"/snapshot_%03d.WWpidsortindex.%i.hdf" %(snap,i),"w")
		hdffile.create_dataset("pidSortedIndexes",data=pidSortedIndexes[GadFileOffsets[i]:GadFileOffsets[i+1]], compression='gzip', compression_opts=7)
		hdffile.create_dataset("fileOffsets",data=GadFileOffsets)
		hdffile.close()

	#Free the memory to be used again
	del pid
	del pidSortedIndexes

	print(Rank,"Done creating file for snapshot",snap,"in",time.time()-start)

