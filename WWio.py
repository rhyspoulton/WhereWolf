import numpy as np 
import h5py
import struct
import os
import mmap
import sys
import time
import multiprocessing as mp
import os


def GetHaloParticles(grpfile,pfile,upfile,HaloIndx):
	"""
	
	Function to extract the partcies from the VELOCIraptor halos
	
	"""

	#Extract the offsets and how many particles the halo has
	offset=np.uint64(grpfile["Offset"][HaloIndx:HaloIndx+2])
	unboffset=np.uint64(grpfile["Offset_unbound"][HaloIndx:HaloIndx+2])
	npart=grpfile["Group_Size"][HaloIndx]

	particleIDs=np.zeros(npart,dtype=np.int64)

	#Update the offset if we are at the end of the file
	if(len(offset)==1):
		offset=[offset[0],pfile["Num_of_particles_in_groups"][:][0]]
	if(len(unboffset)==1):
		unboffset=[unboffset[0],upfile["Num_of_particles_in_groups"][:][0]]

	#Put the particleID's into the array
	numunbound=unboffset[1]-unboffset[0]
	numbound=npart-numunbound
	particleIDs[:numbound]=np.int64(pfile["Particle_IDs"][offset[0]:offset[1]])
	particleIDs[numbound:]=np.int64(upfile["Particle_IDs"][unboffset[0]:unboffset[1]])

	return particleIDs

def ReadGadgetPosVel(Rank,size,GadFilename,ExtractParticleIDs,Gadnumfiles,GadfileMins,GadfileMaxs):

	start = time.time()

	Npart = len(ExtractParticleIDs)
	newPartIDs = np.zeros(Npart,dtype=np.int64)
	pos = np.zeros([Npart,3],dtype=np.float32)
	vel = np.zeros([Npart,3],dtype=np.float32)


	ifiles=np.arange(0,Gadnumfiles,1,dtype=int)

	ifiles = np.roll(ifiles,int(np.floor(Rank * Gadnumfiles/size)))


	for i in ifiles:

		# print(Rank,"is doing file",i)


		GadData = h5py.File(GadFilename+".%i.hdf5" %i,"r")

		pid = np.asarray(GadData["PartType1"]["ParticleIDs"])


		# print(np.min(ExtractParticleIDs),np.max(ExtractParticleIDs),GadfileMins[i],GadfileMaxs[i])
		if np.all((GadfileMins[i]>pid) | (pid>GadfileMaxs[i])):
			print(i)
			continue


		LoadBool = np.in1d(pid,ExtractParticleIDs)
		LoadBool2 = np.zeros([pid.shape[0],3],dtype=bool)
		LoadBool2[LoadBool,:]  = True

		NumLoad = np.sum(LoadBool)

		SetData = np.zeros(NumLoad,dtype="uint64")
		for i,ID in enumerate(pid[LoadBool]):
			SetData[i]=np.where(ExtractParticleIDs==ID)[0][0]

		

		newPartIDs[SetData] = GadData["PartType1"]["ParticleIDs"][LoadBool]
		pos[SetData,:] = GadData["PartType1"]["Coordinates"][LoadBool2].reshape(NumLoad,3)
		vel[SetData,:] = GadData["PartType1"]["Velocities"][LoadBool2].reshape(NumLoad,3)

		GadData.close()


	print("Loaded ID's using in1d in",time.time()-start)

	return pos,vel

def GetParticleData(comm,Rank,size,opt,isnap,trackIndx,tracknpart,GadHeaderInfo,VELnumfiles,VELfilenumhalos,pfiles,upfiles,grpfiles,newPartOffsets,nextpids):

	if(opt.iverbose):
		if(Rank==0): print("Loading in the halo particles from the gadget file(s)")
	start = time.time()

	if(trackIndx.size!=0):
		npartExtractTot = np.sum(tracknpart,dtype=np.int64)
		ExtractParticleIDs = np.zeros(npartExtractTot,dtype=np.uint64)
		PartOffsets = np.zeros(trackIndx.size,dtype=np.uint64)

		partStart = np.uint64(0)
		partEnd = np.uint64(0)

		fileno=0
		offset=0

		for i,Indx in enumerate(trackIndx):

			partStart = partEnd

			partEnd += tracknpart[i]

			#Loop through the data until get the correct file and open that file
			while((Indx+1)>(offset + VELfilenumhalos[fileno])):
				offset+=VELfilenumhalos[fileno]
				fileno+=1

			ExtractParticleIDs[partStart:partEnd] = GetHaloParticles(grpfiles[fileno],pfiles[fileno],upfiles[fileno],int(Indx-offset))
			PartOffsets[i] = partStart

		if(nextpids is not None):
			numNextpids = np.int64(len(nextpids))


			allExtractParticleIDs=np.empty(numNextpids+ExtractParticleIDs.size,dtype="uint64")

			allExtractParticleIDs[:numNextpids] = nextpids
			allExtractParticleIDs[numNextpids:] = ExtractParticleIDs

			del nextpids
			del ExtractParticleIDs

			numNewOffsets = np.int64(len(newPartOffsets))

			allPartOffsets=np.empty(numNewOffsets+PartOffsets.size,dtype="uint64")

			PartOffsets+=numNextpids

			allPartOffsets[:numNewOffsets] = newPartOffsets
			allPartOffsets[numNewOffsets:] = PartOffsets

			del newPartOffsets
			del PartOffsets

			npartExtractTot+=numNextpids

		else:
			allExtractParticleIDs = ExtractParticleIDs
			allPartOffsets = PartOffsets

	else:
		npartExtractTot = nextpids.size
		allExtractParticleIDs = nextpids
		allPartOffsets = newPartOffsets


	# # print("orig",allExtractParticleIDs)
	origIDs=allExtractParticleIDs

	#Make the ID's unique and sorted while returning the inverse and set the number to be extracted
	allExtractParticleIDs, inverseIndxes = np.unique(allExtractParticleIDs,return_inverse=True)
	npartExtract=allExtractParticleIDs.size

	# #Boolean to extract what indexes is needed from the WhereWolf part sorted Indexes file
	# Sel = np.zeros(GadHeaderInfo["TotNpart"],dtype=bool)
	# Sel[allExtractParticleIDs-1] = True

	if(GadHeaderInfo["NumFiles"]==1):

		#First check the file exists
		if((Rank==0) & (os.path.isfile(opt.WWPIDSortedIndexList[opt.Snapshot_offset + isnap]+".WWpidsortindex.hdf")==False)):
			print("The partSortedIndex file",opt.WWPIDSortedIndexList[opt.Snapshot_offset + isnap]+".WWpidsortindex.hdf","not found")
			print("Terminating")
			comm.Abort()

		#Open up the file
		WWPartSortedFile = h5py.File(opt.WWPIDSortedIndexList[opt.Snapshot_offset + isnap]+".WWpidsortindex.hdf","r")

		#Find the locations need to extact the indexes
		LoadBool = np.zeros(GadHeaderInfo["TotNpart"],dtype=bool)
		LoadBool[allExtractParticleIDs - GadHeaderInfo["DMPartIDOffset"]]  =True

		#Extract the index from the file
		partLoc = WWPartSortedFile["pidSortedIndexes"][LoadBool]

		WWPartSortedFile.close()

		#Make it unique so the ID's can loaded and return inverse
		partLoc,inverse = np.unique(partLoc,return_inverse=True)

		#Set the locations need to extract
		LoadBool = np.zeros(GadHeaderInfo["TotNpart"],dtype=bool)
		LoadBool2 = np.zeros([GadHeaderInfo["TotNpart"],3],dtype=bool)
		LoadBool[partLoc] = True
		LoadBool2[partLoc,:] = True

		#Setup the arrays to load the data into
		# newPartIDs = np.zeros(npartExtract,dtype=np.int64)
		pos = np.zeros([npartExtract,3],dtype=np.float32)
		vel = np.zeros([npartExtract,3],dtype=np.float32)

		#Lets check the gadget file exists
		if((Rank==0) & (os.path.isfile(opt.GadFileList[opt.Snapshot_offset + isnap]+".hdf5")==False)):
			print("The Gadget snapshot file",opt.GadFileList[opt.Snapshot_offset + isnap]+".hdf5","not found")
			print("Terminating")
			comm.Abort()

		GadFile = h5py.File(opt.GadFileList[opt.Snapshot_offset + isnap]+".hdf5","r")

		#Load in the desired particles
		# newPartIDs = GadFile["PartType1"]["ParticleIDs"][LoadBool]
		pos[:] = GadFile["PartType1"]["Coordinates"][LoadBool2].reshape(npartExtract,3)
		vel[:] = GadFile["PartType1"]["Velocities"][LoadBool2].reshape(npartExtract,3)

		GadFile.close()

		# newPartIDs = newPartIDs[inverse][inverseIndxes]
		pos = pos[inverse][inverseIndxes]*GadHeaderInfo["Scalefactor"]/GadHeaderInfo["h"]
		vel = vel[inverse][inverseIndxes]*np.sqrt(GadHeaderInfo["Scalefactor"])
		allExtractParticleIDs = allExtractParticleIDs[inverseIndxes]

	else:
		#Roll the fileno so each thread starts on a different file
		ifiles=np.arange(0,GadHeaderInfo["NumFiles"],1,dtype=int)
		ifiles = np.roll(ifiles,int(np.floor(Rank * GadHeaderInfo["NumFiles"]/size)))

		#Create all of the file offsets
		GadFileOffsets = np.zeros(GadHeaderInfo["NumFiles"]+1,dtype=np.int64)

		partLoc = np.zeros(allExtractParticleIDs.size,dtype=np.int64)

		for i in ifiles:
			#First check the file exists
			if((Rank==0) & (os.path.isfile(opt.WWPIDSortedIndexList[opt.Snapshot_offset + isnap]+".WWpidsortindex.%i.hdf" %i)==False)):
				print("The partSortedIndex file",opt.WWPIDSortedIndexList[opt.Snapshot_offset + isnap]+".WWpidsortindex.%i.hdf" %i,"not found")
				print("Terminating")
				comm.Abort()

			WWPartSortedFile = h5py.File(opt.WWPIDSortedIndexList[opt.Snapshot_offset + isnap]+".WWpidsortindex.%i.hdf" %i,"r")
			GadFileOffsets[i] = WWPartSortedFile.attrs["partOffset"][...]
			GadNumPartFile = WWPartSortedFile.attrs["fileNumPart"][...]

			offset = np.sum(allExtractParticleIDs<GadFileOffsets[i])

			fileLoc = (allExtractParticleIDs[(allExtractParticleIDs>=GadFileOffsets[i]) & (allExtractParticleIDs<GadFileOffsets[i]+GadNumPartFile)] - GadFileOffsets[i] - GadHeaderInfo["DMPartIDOffset"] - 1).astype(np.int64)

			fileNpart = fileLoc.size

			LoadBool = np.zeros(GadNumPartFile,dtype=bool)
			LoadBool[fileLoc] = True

			partLoc[offset:offset+fileNpart] = WWPartSortedFile["pidSortedIndexes"][LoadBool]

			WWPartSortedFile.close()

		#Update the final offset
		GadFileOffsets[-1] = GadHeaderInfo["TotNpart"]
		# if(Rank==0):
		# 	print("Done getting the Sorted Indexes in",time.time()-start)


		#Make it unique so the ID's can loaded and return inverse
		# start = time.time()
		partLoc,inverse = np.unique(partLoc,return_inverse=True)
		# print("Done unique in",time.time()-start)


		# Arrays to store the data to be loaded 
		# newPartIDs = np.zeros(npartExtract,dtype=np.int64)
		pos = np.zeros([npartExtract,3],dtype=np.float32)
		vel = np.zeros([npartExtract,3],dtype=np.float32)


		# start = time.time()

		for i in ifiles:
			# print("Doing fileno",i)

			#Find the offset for this file using the offsets
			offset = np.sum(partLoc<GadFileOffsets[i])

			# Find which particle are within this file
			fileLoc = (partLoc[(partLoc>=GadFileOffsets[i]) & (partLoc<GadFileOffsets[i+1])] - GadFileOffsets[i]).astype(np.int64)

			fileNpart = fileLoc.size

			#Check if there are any to load
			if(fileNpart>0):

				#Create boolean arrays to select the data needed
				# LoadBool = np.zeros(GadFileOffsets[i+1]-GadFileOffsets[i],dtype=bool)
				LoadBool2 = np.zeros([GadFileOffsets[i+1]-GadFileOffsets[i],3],dtype=bool)
				# LoadBool[fileLoc] = True
				LoadBool2[fileLoc,:] = True

				#Lets check the gadget file exists
				if((Rank==0) & (os.path.isfile(opt.GadFileList[opt.Snapshot_offset + isnap]+".%i.hdf5" %i)==False)):
					print("The Gadget snapshot file",opt.GadFileList[opt.Snapshot_offset + isnap]+".%i.hdf5" %i,"not found")
					print("Terminating")
					comm.Abort()

				GadFile = h5py.File(opt.GadFileList[opt.Snapshot_offset + isnap]+".%i.hdf5" %i,"r")

				#Extract the data from the file
				# newPartIDs[offset:offset+fileNpart] = GadFile["PartType1"]["ParticleIDs"][LoadBool]
				pos[offset:offset+fileNpart,:] = GadFile["PartType1"]["Coordinates"][LoadBool2].reshape(fileNpart,3)
				vel[offset:offset+fileNpart,:] = GadFile["PartType1"]["Velocities"][LoadBool2].reshape(fileNpart,3)

				GadFile.close()

		

		#Convert to VELOCIraptor (physical uits)
		# newPartIDs = newPartIDs[inverse][inverseIndxes]
		pos = pos[inverse][inverseIndxes]*GadHeaderInfo["Scalefactor"]/GadHeaderInfo["h"]
		vel = vel[inverse][inverseIndxes]*np.sqrt(GadHeaderInfo["Scalefactor"])
		allExtractParticleIDs = allExtractParticleIDs[inverseIndxes]

	# print(np.all(newPartIDs==origIDs))
	
	return allExtractParticleIDs,pos,vel,allPartOffsets


def GetGadFileInfo(comm,GadFileBaseName):



	GadHeaderInfo = {}

	#Lets check if this snapshot is a single file or multiple (mpi split)
	GadFileName =GadFileBaseName + ".hdf5"
	if(os.path.isfile(GadFileName)==False):
		GadFileName = GadFileBaseName + ".0.hdf5"
		if(os.path.isfile(GadFileName)== False):
			print("Could not find file",GadFileName)
			comm.Abort()

	#Open up the first file to see if and how many files the data is split across
	GadFile = h5py.File(GadFileName,"r")

	GadHeaderInfo["DMPartIDOffset"] = np.uint64(GadFile["Header"].attrs["NumPart_Total"][0])
	GadHeaderInfo["TotNpart"] = np.uint64(GadFile["Header"].attrs["NumPart_Total"][1])
	GadHeaderInfo["BoxSize"] = GadFile["Header"].attrs["BoxSize"]
	GadHeaderInfo["partMass"] = GadFile["Header"].attrs["MassTable"][1] # dark matter mass
	GadHeaderInfo["h"] = GadFile["Header"].attrs["HubbleParam"]
	GadHeaderInfo["NumFiles"] = int(GadFile["Header"].attrs["NumFilesPerSnapshot"])
	GadHeaderInfo["Scalefactor"] = GadFile["Header"].attrs["Time"]
	GadHeaderInfo["Redshift"] = GadFile["Header"].attrs["Redshift"]

	if("NumPart_Total_HighWord" in GadFile["Header"].attrs):
		#Lets see if it is populated
		if(np.sum(GadFile["Header"].attrs["NumPart_Total_HighWord"])>0):
			#Find the base and the power for the number of particles
			sel = np.uint64(np.where(GadFile["Header"].attrs["NumPart_Total_HighWord"]>0)[0][0])
			base = np.uint64(GadFile["Header"].attrs["NumPart_Total_HighWord"][sel])
			power = np.uint64(32 + sel)

			#Add this number to the current TotNpart
			GadHeaderInfo["TotNpart"] += np.power(base,power)

  
	GadFile.close()


	return GadHeaderInfo


def GetGadFileMinMax(comm,Rank,size,GadFilename,GadHeaderInfo):

	start=time.time()

	numFilesPerCPU = int(np.floor(GadHeaderInfo["NumFiles"]/size))

	ifilestart = int(numFilesPerCPU*Rank)

	if(Rank==size-1):
		ifileend = GadHeaderInfo["NumFiles"]
	else:
		ifileend = ifilestart+numFilesPerCPU


	localMins = np.zeros(ifileend-ifilestart,dtype="uint64")
	localMaxs = np.zeros(ifileend-ifilestart,dtype="uint64")


	i = 0
	for ifile in range(ifilestart,ifileend):

		GadFile = h5py.File(GadFilename + ".%i.hdf5" %ifile,"r")

		pid = np.asarray(GadFile["PartType1"]["ParticleIDs"])

		localMins[i] = np.min(pid)
		localMaxs[i] = np.max(pid)
		i+=1

		GadFile.close()

	sendcounts = np.array(comm.gather(len(localMins), 0))

	if(Rank==0):
		AllMins = np.empty(sum(sendcounts), dtype=np.uint64)
		AllMaxs = np.empty(sum(sendcounts), dtype=np.uint64)
	else:
		AllMins=None
		AllMaxs=None

	comm.Gatherv(sendbuf=localMins, recvbuf=(AllMins, sendcounts), root=0)
	comm.Gatherv(sendbuf=localMaxs, recvbuf=(AllMaxs, sendcounts), root=0)


	AllMins=comm.bcast(AllMins,root=0)
	AllMaxs=comm.bcast(AllMaxs,root=0)

	print("Done in",time.time()-start)

	return AllMins,AllMaxs


def AdjustforPeriod(snapdata):
    """
    Map halo positions from 0 to box size
    """

    boxval = snapdata["SimulationInfo"]["Period"]
    wdata = np.where(snapdata["Xc"] < 0)
    snapdata["Xc"][wdata] += boxval
    wdata = np.where(snapdata["Yc"] < 0)
    snapdata["Yc"][wdata] += boxval
    wdata = np.where(snapdata["Zc"] < 0)
    snapdata["Zc"][wdata] += boxval

    wdata = np.where(snapdata["Xc"] > boxval)
    snapdata["Xc"][wdata] -= boxval
    wdata = np.where(snapdata["Yc"] > boxval)
    snapdata["Yc"][wdata] -= boxval
    wdata = np.where(snapdata["Zc"] > boxval)
    snapdata["Zc"][wdata] -= boxval



def ReadVELOIraptorCatalogueNpartSplit(basefilename,ihalostart,ihaloend,filenumhalos,iverbose=0):

	name = mp.current_process().name

	if(int(ihaloend-ihalostart)==0):
		return []

	if(iverbose): print("Reading VELOCIraptor properties file",basefilename)

	filename=basefilename +".properties.0"
	
	if (os.path.isfile(filename)==False):
		raise SystemExit("%s not found" %filename)

	offset = 0
	fileno = 0


	while((ihalostart+1)>(offset + filenumhalos[fileno])):
		offset+=filenumhalos[fileno]
		fileno+=1

	ifilestart = ihalostart-np.sum(filenumhalos[:fileno])
	inumhalos = ihaloend - ihalostart  
	ioffset = np.int64(0)
	extractednumhalos = np.int64(0)

	npart = np.zeros(inumhalos,dtype=np.int32)

	while(extractednumhalos<inumhalos):

		# If we need to extract the whole file or not
		if((inumhalos-extractednumhalos)>(filenumhalos[fileno]-ifilestart)):
			ifileend = filenumhalos[fileno]
		else:
			ifileend = inumhalos - extractednumhalos

		filename=basefilename+".properties." + str(fileno)

		extractednumhalos+=ifileend - ifilestart 
		# print(name,"is trying to read from",ifilestart,"to",ifileend,"where the file contains",filenumhalos[fileno],"halos with",ioffset,extractednumhalos,"required")

		hdffile = h5py.File(filename,"r")
		if(npart[ioffset:extractednumhalos].shape[0]!=hdffile["npart"][ifilestart:ifileend].shape[0]):
			print(name,ioffset,extractednumhalos,inumhalos,ifilestart,ifileend,filenumhalos[fileno])

		npart[ioffset:extractednumhalos] = hdffile["npart"][ifilestart:ifileend]

		hdffile.close()

		ioffset+=ifileend - ifilestart 

		fileno+=1

		ifilestart=0

	
	return npart


def ReadVELOCIraptorTreeDescendantSplit(basefilename,ihalostart,ihaloend,iverbose=0):

	if(int(ihaloend-ihalostart)==0):
		return []

	if(iverbose): print("Reading VELOCIraptor tree file",basefilename)
	filename=basefilename +".tree"

	if (os.path.isfile(filename)==False):
		raise SystemExit("%s not found" %filename)

	tree = {"NumDesc": [], "Descendants": [], "Rank": []}

	with h5py.File(filename,"r") as hdffile:

		tree["NumDesc"] = np.asarray(hdffile["NumDesc"][ihalostart:ihaloend])

		#See if the dataset exits
		if("DescOffsets" in hdffile.keys()):


			#Get the indices for the main descedant
			Offsets = np.asarray(hdffile["DescOffsets"][ihalostart:ihaloend])

			#Lets remove the offsets where the halo has no descedant
			descSel = np.where(tree["NumDesc"]>0)[0]
			Offsets = Offsets[descSel]

			#Create a Bool array to select the main descedants
			mainDescSel = np.zeros(hdffile["Descendants"].size,dtype=bool)
			mainDescSel[Offsets] = True

			#Delcare arrays for descendats and thier ranks
			tree["Descendants"] = np.zeros(ihaloend-ihalostart,dtype=np.uint64)
			tree["Rank"] = -1*np.ones(ihaloend-ihalostart,dtype=np.int32)


			# Read in the data splitting it up as reading it in
			tree["Descendants"][descSel] = hdffile["Descendants"][mainDescSel]
			tree["Rank"][descSel] = hdffile["Ranks"][mainDescSel]
	return tree


def ReadVELOCIraptorTreeDescendant(comm,basefilename,iverbose=0):


	if(iverbose): print("Reading VELOCIraptor tree file",basefilename)
	filename=basefilename +".tree"

	if (os.path.isfile(filename)==False):
		print("TreeFrog file %s not found" %filename)
		comm.Abort()

	tree = {}
	#Store the options this tree was built with
	treeOpt={}

	with h5py.File(filename,"r") as hdffile:

		#Extract the tree construction options
		treeOpt["Core_fraction"]=hdffile.attrs["Core_fraction"][...]
		treeOpt["Core_min_number_of_particles"]=hdffile.attrs["Core_min_number_of_particles"][...]
		treeOpt["Merit_limit"]=hdffile.attrs["Merit_limit"][...]
		treeOpt["Merit_limit_for_next_step"]=hdffile.attrs["Merit_limit_for_next_step"][...]
		treeOpt["Number_of_steps"]=hdffile.attrs["Number_of_steps"][...]
		if("Merit_type" in hdffile.attrs.keys()):
			treeOpt["Merit_type"]=hdffile.attrs["Merit_type"][...]
		else:
			print("Merit type not found in the TreeFrog header setting it as MERITRankWeightedBoth(6)")
			treeOpt["Merit_type"]=6

		#See if the dataset exits
		if("DescOffsets" in hdffile.keys()):


			tree["NumDesc"] = np.asarray(hdffile["NumDesc"])

			#Get the indices for the main descedant
			Offsets = np.asarray(hdffile["DescOffsets"])

			#Lets remove the offsets where the halo has no descedant
			descSel = np.where(tree["NumDesc"]>0)[0]
			Offsets = Offsets[descSel]

			#Create a Bool array to select the main descedants
			mainDescSel = np.zeros(hdffile["Descendants"].size,dtype=bool)
			mainDescSel[Offsets] = True

			#Delcare arrays for descendats and thier ranks
			tree["Descen"] = np.zeros(tree["NumDesc"].size,dtype=np.uint64)
			tree["Rank"] = -1*np.ones(tree["NumDesc"].size,dtype=np.int32)
			tree["Merits"] = np.zeros(tree["NumDesc"].size,dtype=np.float32)


			# Read in the data splitting it up as reading it in
			tree["Descen"][descSel] = hdffile["Descendants"][mainDescSel]
			tree["Rank"][descSel] = hdffile["Ranks"][mainDescSel]
			tree["Merits"][descSel] =  hdffile["Merits"][mainDescSel]

			del Offsets


	return treeOpt,tree

def SetupParallelIO(comm,opt,nprocess):

	print("Setting up for ParallelIO")

	#Set the number of halos per cpu	
	ihalostart = np.zeros([nprocess,opt.numsnaps],dtype=np.int64)
	ihaloend = np.zeros([nprocess,opt.numsnaps],dtype=np.int64)
	numhalos = np.zeros(opt.numsnaps,dtype=np.uint64)


	for snap in range(opt.numsnaps):

		#Lets load the halos from the properties file
		filename= opt.VELFileList[snap] + ".properties.0"

		if(os.path.isfile(filename)==False):
			print("VELOCIraptor file",filename,"not found")
			print("WhereWolf only works when VELOCIraptor has been run with MPI and number of threads > 1")
			comm.Abort()
		try:
			hdffile = h5py.File(filename,"r")
		except OSError:
			print("WhereWolf only works with hdf5 output of VELOCIraptor")
			comm.Abort()

		numhalos[snap] = np.uint64(hdffile["Total_num_of_groups"][0])

		hdffile.close()

		#Lets set the number of halos per cpu-
		numPerCPU = np.floor(numhalos[snap]/np.float(nprocess))
		istart = 0
		iend = numPerCPU

		for i in range(nprocess):

			ihalostart[i,snap] = istart
			ihaloend[i,snap] = iend

			istart=iend
			iend+=numPerCPU
			

		ihaloend[-1,snap] = numhalos[snap]


	return ihalostart, ihaloend, numhalos


def OpenVELOCIraptorFiles(filename):
	"""

	Function to open up the VELOCIraptor files for a snapshot

	"""

	#First open up the first property file to read the number of files
	catfilename=filename+".catalog_groups.0"
	catfile=h5py.File(catfilename,"r")
	numfiles = int(catfile["Num_of_files"][:])
	numhalos = np.zeros(numfiles+1,dtype=np.int64)
	catfile.close()

	#Setup list to store the file pointers
	pfiles = [0 for i in range(numfiles) ]
	upfiles = [0 for i in range(numfiles)]
	grpfiles = [0 for i in range(numfiles)]


	#Loop over all the files opening them and storing them in the pointer list
	for fileno in range(numfiles):

		pfilename=filename+".catalog_particles."+str(fileno)
		pfiles[fileno]=h5py.File(pfilename,"r")
		upfilename=filename+".catalog_particles.unbound."+str(fileno)
		upfiles[fileno]=h5py.File(upfilename,"r")
		grpfilename=filename+".catalog_groups."+str(fileno)
		grpfiles[fileno]=h5py.File(grpfilename,"r")
		numhalos[fileno]=np.uint64(grpfiles[fileno]["Num_of_groups"][0])

	return numhalos,numfiles,pfiles,upfiles, grpfiles


def CloseVELOCIraptorFiles(filename,numfiles,pfiles,upfiles,grpfiles):
	"""

	Function to close the VELOCIraptor files

	"""

	for fileno in range(numfiles):
		pfiles[fileno].close()
		upfiles[fileno].close()
		grpfiles[fileno].close() 


def ExtractGadgetpartIDs(GadFileBaseName,inumfiles,pid,fileoffset):

	name = mp.current_process().name

	print(pid.shape)

	# Loaded all of the particle IDs and the number of particles in each file
	for i in inumfiles:
		# print(name,"Doing fileno",i)

		GadFile = h5py.File(GadFileBaseName+".%i.hdf5" %i,"r")

		offset1 = fileoffset[i]
		offset2 = fileoffset[i+1]
		print(name,offset1,offset2,offset2-offset1,GadFile["PartType1"]["ParticleIDs"][:].shape,pid[offset1:offset2].shape)

		pid[offset1:offset2] = GadFile["PartType1"]["ParticleIDs"][:]

		GadFile.close()


def ReadGadgethdf(GadFileBaseName,partIDs,pid,numfiles,fileoffset):


	print(partIDs)

	Npart = len(partIDs)

	#Sort the IDs only where we need them to be sorted and then extract them
	start = time.time()
	partLoc = np.take(np.argpartition(pid,partIDs-1),partIDs-1)
	print("Done argpartition of",Npart,"values in",time.time()-start)


	#Make it unique so the ID's can loaded and return inverse
	# start = time.time()
	partLoc,inverse = np.unique(partLoc,return_inverse=True)
	# print("Done unique in",time.time()-start)

	# Arrays to store the data to be loaded 
	# newPartIDs = np.zeros(Npart,dtype=np.int64)
	pos = np.zeros([Npart,3],dtype=np.float32)
	vel = np.zeros([Npart,3],dtype=np.float32)



	offset = 0

	# start = time.time()

	for i in range(numfiles):
		# print("Doing fileno",i)

		# Find which particle are within this file
		fileLoc = (partLoc[(partLoc>=fileoffset[i]) & (partLoc<fileoffset[i+1])] - fileoffset[i]).astype(int)


		fileNpart = len(fileLoc)

		#Check if there are any to load
		if(fileNpart>0):

			#Create boolean arrays to select the data needed
			LoadBool = np.zeros(fileoffset[i+1]-fileoffset[i],dtype=bool)
			LoadBool2 = np.zeros([fileoffset[i+1]-fileoffset[i],3],dtype=bool)
			LoadBool[fileLoc] = True
			LoadBool2[fileLoc,:] = True

			GadFile = h5py.File(GadFileBaseName+".%i.hdf5" %i,"r")

			#Extract the data from the file
			# newPartIDs[offset:offset+fileNpart] = GadFile["PartType1"]["ParticleIDs"][LoadBool]
			pos[offset:offset+fileNpart,:] = GadFile["PartType1"]["Coordinates"][LoadBool2].reshape(fileNpart,3)
			vel[offset:offset+fileNpart,:] = GadFile["PartType1"]["Velocities"][LoadBool2].reshape(fileNpart,3)

			GadFile.close()

			offset+=fileNpart

	# newPartIDs = newPartIDs[inverse]
	pos = pos[inverse]
	vel = vel[inverse]

	# print("done extracting the data in",time.time()-start)

	# print(newPartIDs[:5])
	return pos,vel

def OutputWhereWolfTreeData(opt,snap,appendTreeData,updateTreeData,HALOIDVAL=1000000000000):

	treeFields=["ID","NumDesc"]

	#Open up the TreeFrog tree file
	treefile = h5py.File(opt.TreeFileList[snap] + ".tree","r")

	#Open up a .WW file to output all the tree data
	WWtreefile = h5py.File(opt.outputdir+"/snapshot_%03d.VELOCIraptor.WW.tree" %(snap),"w")

	#Give the WW treefile the same header infor as the treefrog file
	for attrsField in treefile.attrs.keys():
		WWtreefile.attrs[attrsField] = treefile.attrs[attrsField]

	#Add the header info that this is a WWfile
	WWtreefile.attrs["WWfile"] = 1
	# Do the Descendants dataset as this needs to be updated if it exists
	if(("Descendants" in treefile.keys()) & (len(updateTreeData["ID"])>0)):

		#Find the location of the halos need to update
		updateIndexes = (updateTreeData["ID"]%HALOIDVAL-1).astype(int)

		#Find the location of thier direct descendants from the offsets
		TFDescOffsets = np.asarray(treefile["DescOffsets"])
		descIndexes = TFDescOffsets[updateIndexes]

		#Now load in the descendant data
		TFDescen = np.asarray(treefile["Descendants"])
		dsetSize = TFDescen.size

		# Update the descen to point to the WWhalo
		TFDescen[descIndexes] = updateTreeData["Descendants"]

		#Now also update the Ranks
		TFRanks = np.asarray(treefile["Ranks"])

		# Update the Ranks so the connections with the WW halos are Rank 0
		TFRanks[descIndexes] = 0

		#Then the updated merits
		TFMerits = np.asarray(treefile["Merits"])

		#Update the merits to the ones calculated
		TFMerits[descIndexes] = updateTreeData["Merits"]

		#Done with the updateTreeData
		del updateTreeData
	elif("DescOffsets" in treefile.keys()):

		#Find the location of thier direct descendants from the offsets
		TFDescOffsets = np.asarray(treefile["DescOffsets"])
		#Now load in the descendant data
		TFDescen = np.asarray(treefile["Descendants"])
		TFRanks = np.asarray(treefile["Ranks"])
		TFMerits = np.asarray(treefile["Merits"])
		dsetSize = TFRanks.size

	#Check if there is anything to append to the tree
	if(len(appendTreeData["ID"])>0):


		if("Descendants" in treefile.keys()):

			sel = appendTreeData["Descendants"]>0

			#Check the number to append
			Nappend = np.sum(sel)

			#Allocate an array to store all of the descendants
			allTreeDescen = np.empty(dsetSize + Nappend,dtype=TFDescen.dtype)

			#Insert the data into the array
			allTreeDescen[:dsetSize] = TFDescen
			allTreeDescen[dsetSize:] = appendTreeData["Descendants"][sel]

			#Output this to the WWtreefile
			WWtreefile.create_dataset("Descendants",data=allTreeDescen)

			#Done with changing the Descendants dataset
			del allTreeDescen
			del TFDescen

			#Allocate a array to store all the TFdata
			allRanks = np.empty(dsetSize + Nappend,dtype=TFRanks.dtype)

			#Lets insert them into the array of all TFdata
			allRanks[:dsetSize] = TFRanks
			allRanks[dsetSize:] = appendTreeData["Ranks"][sel]

			WWtreefile.create_dataset("Ranks",data=allRanks)


			#Now we are done with the TFdata they can be deleted
			del allRanks
			del TFRanks

			#Allocate a array to store all the TFdata
			allMerits = np.empty(dsetSize + Nappend,dtype=TFMerits.dtype)

			#Lets insert them into the array of all TFdata
			allMerits[:dsetSize] = TFMerits
			allMerits[dsetSize:] = appendTreeData["Merits"][sel]

			WWtreefile.create_dataset("Merits",data=allMerits)


			#Now we are done with the TFdata they can be deleted
			del allMerits
			del TFMerits

			#Check the number to append
			Nappend = appendTreeData["NumDesc"].size


			#Now append the WW descendants offsets
			offset = dsetSize
			WWdescOffsets = np.zeros(Nappend,dtype=np.uint64)

			#Need to build a descenOffset array based on the number of descen
			for i,Ndescen in enumerate(appendTreeData["NumDesc"]):

				WWdescOffsets[i] = offset
				offset += Ndescen

			#Load in the descendant offsets from the tree
			dsetSize = TFDescOffsets.size

			#Allocate an array to store all of the offsets
			allDescOffsets = np.empty(dsetSize + Nappend,dtype =TFDescOffsets.dtype )

			#inset the data into the array
			allDescOffsets[:dsetSize] = TFDescOffsets
			allDescOffsets[dsetSize:] = WWdescOffsets

			#Output to the WW tree file
			WWtreefile.create_dataset("DescOffsets",data=allDescOffsets)

			#Done with the data now it can be deleted
			del allDescOffsets
			del TFDescOffsets



		#Check the number to append
		Nappend = appendTreeData["ID"].size


		#Lets first append the data to the TreeFrog tree
		for field in treeFields:

			# First load in the data from the TF tree
			TFdata = np.asarray(treefile[field])
			dsetSize = TFdata.size

			#Allocate a array to store all the TFdata
			alltreedata = np.empty(dsetSize + Nappend,dtype=TFdata.dtype)

			#Lets insert them into the array of all TFdata
			alltreedata[:dsetSize] = TFdata
			alltreedata[dsetSize:] = appendTreeData[field]

			WWtreefile.create_dataset(field,data=alltreedata)


			#Now we are done with the TFdata they can be deleted
			del alltreedata
			del TFdata

		#### For debugging ####

		# debugdata = np.asarray(treefile["ID"])

		# alldebugdata = np.empty(dsetSize + Nappend,dtype=debugdata.dtype)

		# alldebugdata[:dsetSize] = debugdata
		# alldebugdata[dsetSize:] = appendTreeData["endDesc"]

		# WWtreefile.create_dataset("endDesc",data=alldebugdata)

		# del debugdata
		# del alldebugdata
		
		#Done with the appendTreeData
		del appendTreeData

	else:

		#Output the updated descendants to the WWfile
		WWtreefile.create_dataset("Descendants",data=TFDescen)
		WWtreefile.create_dataset("DescOffsets",data=TFDescOffsets)
		WWtreefile.create_dataset("Ranks",data=TFRanks)
		WWtreefile.create_dataset("Merits",data=TFMerits)

		#Done with the datasets
		del TFDescen
		del TFDescOffsets
		del TFRanks
		del TFMerits


		#Output the rest if the tree data to the WW file
		for field in treeFields:

			#Load in the data from the TF file
			TFdata = np.asarray(treefile[field])

			#Output to the WW tree file
			WWtreefile.create_dataset(field,data=TFdata)

		# ## For debuging
		# debugdata = np.asarray(treefile["ID"])

		# WWtreefile.create_dataset("endDesc",data=debugdata)


	#Close the files
	treefile.close()
	WWtreefile.close()



def AddWhereWolfFileParallel(comm,Rank,size,catfilename,appendData,Nappend,TotNappend):

	if(Rank==0):
		print("Adding",TotNappend,"haloes to the VELOCIraptor files")

	

	ext=[".properties",".catalog_groups",".hierarchy",".catalog_particles",".catalog_particles.unbound"]
	# ext=[".properties",".catalog_groups",".catalog_particles",".catalog_particles.unbound"]
	numgroups=appendData["Num_of_groups"]
	numpart = np.sum(appendData["npart"],dtype=int)
	appendData["Number_of_substructures_in_halo"]=np.zeros(TotNappend)
	for filext in ext:

		datatype={}

		if(Rank==0):
			filename=catfilename+filext+".0"

			halofile=h5py.File(filename,"r")
			numfiles=int(halofile["Num_of_files"][0])


			if((filext==".catalog_particles.unbound") or (filext==".catalog_particles")):  

				Total_num_of_particles_in_all_groups=halofile["Total_num_of_particles_in_all_groups"][0]+ numpart

			elif((filext==".properties") or (filext==".catalog_groups") or (filext==".hierarchy")):

				Total_num_of_groups=halofile["Total_num_of_groups"][0]+TotNappend


			fieldnames = list(halofile.keys())

			for field in fieldnames: datatype[field]=halofile[field].dtype
			halofile.close()

		else:

			fieldnames = None
			datatype = None
			numfiles = None

			if((filext==".catalog_particles.unbound") or (filext==".catalog_particles")):  
				Total_num_of_particles_in_all_groups = None

			elif((filext==".properties") or (filext==".catalog_groups") or (filext==".hierarchy")):
				Total_num_of_groups = None

		#Done extracting the data from the VELOCIraptor file

		#Now lets broadcast the data to the other processes
		fieldnames = comm.bcast(fieldnames,root=0)
		datatype = comm.bcast(datatype,root=0)
		numfiles = comm.bcast(numfiles,root=0)


		if((filext==".catalog_particles.unbound") or (filext==".catalog_particles")):  
			Total_num_of_particles_in_all_groups = comm.bcast(Total_num_of_particles_in_all_groups,root=0)

		elif((filext==".properties") or (filext==".catalog_groups") or (filext==".hierarchy")):
			Total_num_of_groups = comm.bcast(Total_num_of_groups,root=0)



		#Update the appendData to include the data from the the other VELOCIraptor halos
		if((filext==".catalog_particles.unbound") or (filext==".catalog_particles")):  

				if(filext[-7:]=="unbound"):
					appendData["Num_of_particles_in_groups"]=0
				else:
					appendData["Num_of_particles_in_groups"]=numpart

				appendData["Total_num_of_particles_in_all_groups"]=Total_num_of_particles_in_all_groups
				#appendData["Particle_types"]=np.ones(appendData["Num_of_particles_in_groups"])

		elif((filext==".properties") or (filext==".catalog_groups") or (filext==".hierarchy")):

			appendData["Total_num_of_groups"]=Total_num_of_groups

		appendData["Num_of_files"]=numfiles+size
		appendData["File_id"]=numfiles + Rank

			
		#Write out the WW halos

		WWfilename=catfilename+filext+"."+str(numfiles + Rank)

		halofile=h5py.File(WWfilename,"w")

		halofile.attrs["WWfile"]=1
		
		for key in fieldnames:
			if((key=="Particle_IDs") & (filext[-7:]=="unbound")):
				halofile.create_dataset(key,data=np.zeros(appendData["Particle_IDs_unbound"]),dtype=datatype[key])
			elif(key in appendData.keys()):	
				halofile.create_dataset(key,data=np.array(appendData[key]),dtype=datatype[key])
			else:
				halofile.create_dataset(key,data=-1*np.ones(numgroups),dtype=datatype[key])
		halofile.close()


		if(Rank==0):


			numFilesPerCPU = int(numfiles/size)

			ifilestart = 0
			ifileend =  numFilesPerCPU

			processfilestart = ifilestart
			processfileend = ifileend


			for iprocess in range(1,size):



				processfilestart = processfileend 
				processfileend += numFilesPerCPU

				comm.send(processfilestart,dest=iprocess,tag=17)

				if(iprocess==size-1):
					comm.send(numfiles,dest=iprocess,tag=26)
				else:
					comm.send(processfileend,dest=iprocess,tag=26)
		
		else:
			ifilestart=comm.recv(source=0,tag=17)
			ifileend = comm.recv(source=0,tag=26)

	



		#Now update the data ine the VELOCIraptor files to include the WW halos 
		for j in range(ifilestart,ifileend):
			filename=catfilename+filext+"."+str(j)
			# print(filename)

			halofile=h5py.File(filename,"r+")

			halofile["Num_of_files"][...]=numfiles+size

			if((filext==".catalog_particles.unbound") or (filext==".catalog_particles")):  
				halofile["Total_num_of_particles_in_all_groups"][...]=Total_num_of_particles_in_all_groups

			elif((filext==".properties") or (filext==".catalog_groups") or (filext==".hierarchy")):
				halofile["Total_num_of_groups"][...]=Total_num_of_groups

			halofile.close()


def AddWhereWolfFile(catfilename,appendData):

	print("Adding the data to the VELOCIraptor files")
	

	ext=[".properties",".catalog_groups",".hierarchy",".catalog_particles",".catalog_parttypes",".catalog_particles.unbound",".catalog_parttypes.unbound"]
	# ext=[".properties",".catalog_groups",".catalog_particles",".catalog_particles.unbound"]
	numgroups=appendData["Num_of_groups"]
	appendData["Number_of_substructures_in_halo"]=np.zeros(numgroups)
	for filext in ext:

		datatype={}
		filename=catfilename+filext+".0"

		halofile=h5py.File(filename,"r+")
		numfiles=int(halofile["Num_of_files"][:])


		if((filext==".catalog_particles.unbound") or (filext==".catalog_particles") or (filext==".catalog_parttypes.unbound") or (filext==".catalog_parttypes")):  

			if(filext[-7:]=="unbound"):
				appendData["Num_of_particles_in_groups"]=0
			else:
				appendData["Num_of_particles_in_groups"]=len(appendData["Particle_IDs"])

			numpart=appendData["Num_of_particles_in_groups"]
			Total_num_of_particles_in_all_groups=halofile["Total_num_of_particles_in_all_groups"][:]+ numpart
			appendData["Total_num_of_particles_in_all_groups"]=Total_num_of_particles_in_all_groups
			appendData["Particle_types"]=np.ones(appendData["Num_of_particles_in_groups"])

		elif((filext==".properties") or (filext==".catalog_groups") or (filext==".hierarchy")):

			Total_num_of_groups=halofile["Total_num_of_groups"][:]+numgroups
			appendData["Total_num_of_groups"]=Total_num_of_groups

		appendData["Num_of_files"]=numfiles+1
		appendData["File_id"]=numfiles


		fieldnames = list(halofile.keys())

		for field in fieldnames: datatype[field]=halofile[field].dtype
		halofile.close()

		for j in range(numfiles):
			filename=catfilename+filext+"."+str(j)

			halofile=h5py.File(filename,"r+")

			halofile["Num_of_files"][:]=numfiles+1

			if((filext==".catalog_particles.unbound") or (filext==".catalog_particles") or (filext==".catalog_parttypes.unbound") or (filext==".catalog_parttypes")):  
				halofile["Total_num_of_particles_in_all_groups"][:]=Total_num_of_particles_in_all_groups

			elif((filext==".properties") or (filext==".catalog_groups") or (filext==".hierarchy")):
				halofile["Total_num_of_groups"][:]=Total_num_of_groups

			halofile.close()
			
		

		WWfilename=catfilename+filext+"."+str(numfiles)

		halofile=h5py.File(WWfilename,"w")

		halofile.attrs["WWfile"]=1

		
		for key in fieldnames:
			if((key=="Particle_IDs") & (filext[-7:]=="unbound")):
				halofile.create_dataset(key,data=np.array(appendData["Particle_IDs_unbound"]),dtype=datatype[key])
			elif(key in appendData.keys()):	
				halofile.create_dataset(key,data=np.array(appendData[key]),dtype=datatype[key])
			else:
				halofile.create_dataset(key,data=-1*np.ones(numgroups),dtype=datatype[key])
		halofile.close()




def Reset_Files(VELFilename):
	"""
	Function to reset the VELOCIraptor files and remove the WhereWolf file from the catalogue

	"""

	ext=[".properties",".catalog_groups",".hierarchy",".catalog_particles",".catalog_particles.unbound"]
	# ext=[".properties",".catalog_groups",".catalog_particles",".catalog_particles.unbound"]

	print("Reseting files",VELFilename)

	#Loop over all the files
	for filext in ext:
		filename=VELFilename+filext+".0"
		halofile=h5py.File(filename,"r+")
		numfiles=int(halofile["Num_of_files"][:])
		halofile.close()

		#Recusively remove the WW files
		for ifile in range(numfiles-1,-1,-1):
			#Open up final file and check if it is a WhereWolf file
			WWfilename=VELFilename+filext+"."+str(ifile)
			WWfile=h5py.File(WWfilename,"r")
			try:
				WWfile.attrs["WWfile"]
			except KeyError:
				WWfile.close()
				break

			print("Updating the VELOCIraptor files and removing the WW file ",WWfilename)

			#Reset the data in the VELOCIraptor files
			if((filext==".catalog_particles.unbound") or (filext==".catalog_particles")): 
				new_Total_num_of_particles_in_all_groups=WWfile["Total_num_of_particles_in_all_groups"].value - WWfile["Num_of_particles_in_groups"].value

			elif((filext==".properties") or (filext==".catalog_groups") or (filext==".hierarchy")):
				new_Total_num_of_groups=WWfile["Total_num_of_groups"].value - WWfile["Num_of_groups"].value


			numfiles=numfiles - 1 

			WWfile.close()

			for j in range(numfiles):
				filename=VELFilename+filext+"."+str(j)

				halofile=h5py.File(filename)

				halofile["Num_of_files"][()]=numfiles

				if((filext==".catalog_particles.unbound") or (filext==".catalog_particles")):  
					halofile["Total_num_of_particles_in_all_groups"][()]=new_Total_num_of_particles_in_all_groups

				elif((filext==".properties") or (filext==".catalog_groups") or (filext==".hierarchy")):
					halofile["Total_num_of_groups"][()]=new_Total_num_of_groups

				halofile.close()


			os.remove(WWfilename)


def ReadPropertyFile(basefilename,GadHeaderInfo,ibinary=0,iseparatesubfiles=0, desiredfields=[]):
	"""
	VELOCIraptor/STF files in various formats
	for example ascii format contains
	a header with
		filenumber number_of_files
		numhalos_in_file nnumhalos_in_total
	followed by a header listing the information contain. An example would be
		ID(1) ID_mbp(2) hostHaloID(3) numSubStruct(4) npart(5) Mvir(6) Xc(7) Yc(8) Zc(9) Xcmbp(10) Ycmbp(11) Zcmbp(12) VXc(13) VYc(14) VZc(15) VXcmbp(16) VYcmbp(17) VZcmbp(18) Mass_tot(19) Mass_FOF(20) Mass_200mean(21) Mass_200crit(22) Mass_BN97(23) Efrac(24) Rvir(25) R_size(26) R_200mean(27) R_200crit(28) R_BN97(29) R_HalfMass(30) Rmax(31) Vmax(32) sigV(33) veldisp_xx(34) veldisp_xy(35) veldisp_xz(36) veldisp_yx(37) veldisp_yy(38) veldisp_yz(39) veldisp_zx(40) veldisp_zy(41) veldisp_zz(42) lambda_B(43) Lx(44) Ly(45) Lz(46) q(47) s(48) eig_xx(49) eig_xy(50) eig_xz(51) eig_yx(52) eig_yy(53) eig_yz(54) eig_zx(55) eig_zy(56) eig_zz(57) cNFW(58) Krot(59) Ekin(60) Epot(61) n_gas(62) M_gas(63) Xc_gas(64) Yc_gas(65) Zc_gas(66) VXc_gas(67) VYc_gas(68) VZc_gas(69) Efrac_gas(70) R_HalfMass_gas(71) veldisp_xx_gas(72) veldisp_xy_gas(73) veldisp_xz_gas(74) veldisp_yx_gas(75) veldisp_yy_gas(76) veldisp_yz_gas(77) veldisp_zx_gas(78) veldisp_zy_gas(79) veldisp_zz_gas(80) Lx_gas(81) Ly_gas(82) Lz_gas(83) q_gas(84) s_gas(85) eig_xx_gas(86) eig_xy_gas(87) eig_xz_gas(88) eig_yx_gas(89) eig_yy_gas(90) eig_yz_gas(91) eig_zx_gas(92) eig_zy_gas(93) eig_zz_gas(94) Krot_gas(95) T_gas(96) Zmet_gas(97) SFR_gas(98) n_star(99) M_star(100) Xc_star(101) Yc_star(102) Zc_star(103) VXc_star(104) VYc_star(105) VZc_star(106) Efrac_star(107) R_HalfMass_star(108) veldisp_xx_star(109) veldisp_xy_star(110) veldisp_xz_star(111) veldisp_yx_star(112) veldisp_yy_star(113) veldisp_yz_star(114) veldisp_zx_star(115) veldisp_zy_star(116) veldisp_zz_star(117) Lx_star(118) Ly_star(119) Lz_star(120) q_star(121) s_star(122) eig_xx_star(123) eig_xy_star(124) eig_xz_star(125) eig_yx_star(126) eig_yy_star(127) eig_yz_star(128) eig_zx_star(129) eig_zy_star(130) eig_zz_star(131) Krot_star(132) tage_star(133) Zmet_star(134)
	then followed by data
	Note that a file will indicate how many files the total output has been split into
	Not all fields need be read in. If only want specific fields, can pass a string of desired fields like
	['ID', 'Mass_FOF', 'Krot']
	#todo still need checks to see if fields not present and if so, not to include them or handle the error
	"""
	#this variable is the size of the char array in binary formated data that stores the field names
	CHARSIZE=40

	start = time.clock()
	inompi=True
	filename=basefilename+".properties"
	#load header
	if (os.path.isfile(filename)==True):
		numfiles=0
	else:
		filename=basefilename+".properties"+".0"
		inompi=False
		if (os.path.isfile(filename)==False):
			print("file not found")
			return []
	byteoffset=0
	#used to store fields, their type, etc
	fieldnames=[]
	fieldtype=[]
	fieldindex=[]

	if (ibinary==0):
		#load ascii file
		halofile = open(filename, 'r')
		#read header information
		[filenum,numfiles]=halofile.readline().split()
		filenum=int(filenum);numfiles=int(numfiles)
		[numhalos, numtothalos]= halofile.readline().split()
		numhalos=np.uint64(numhalos);numtothalos=np.uint64(numtothalos)
		names = ((halofile.readline())).split()
		#remove the brackets in ascii file names
		fieldnames= [fieldname.split("(")[0] for fieldname in names]
		for i in np.arange(fieldnames.__len__()):
			fieldname=fieldnames[i]
			if fieldname in ["ID","hostHalo","numSubStruct","npart","n_gas","n_star"]:
				fieldtype.append(np.uint64)
			elif fieldname in ["ID_mbp"]:
				fieldtype.append(np.int64)
			else:
				fieldtype.append(np.float64)
		halofile.close()
		#if desiredfields is NULL load all fields
		#but if this is passed load only those fields
		if (len(desiredfields)>0):
			lend=len(desiredfields)
			fieldindex=np.zeros(lend,dtype=int)
			desiredfieldtype=[[] for i in range(lend)]
			for i in range(lend):
				fieldindex[i]=fieldnames.index(desiredfields[i])
				desiredfieldtype[i]=fieldtype[fieldindex[i]]
			fieldtype=desiredfieldtype
			fieldnames=desiredfields
		#to store the string containing data format
		fieldtypestring=''
		for i in np.arange(fieldnames.__len__()):
			if fieldtype[i]==np.uint64: fieldtypestring+='u8,'
			elif fieldtype[i]==np.int64: fieldtypestring+='i8,'
			elif fieldtype[i]==np.float64: fieldtypestring+='f8,'

	elif (ibinary==1):
		#load binary file
		halofile = open(filename, 'rb')
		[filenum,numfiles]=np.fromfile(halofile,dtype=np.int32,count=2)
		[numhalos,numtothalos]=np.fromfile(halofile,dtype=np.uint64,count=2)
		headersize=np.fromfile(halofile,dtype=np.int32,count=1)[0]
		byteoffset=np.dtype(np.int32).itemsize*3+np.dtype(np.uint64).itemsize*2+4*headersize
		for i in range(headersize):
			fieldnames.append(unpack('s', halofile.read(CHARSIZE)).strip())
		for i in np.arange(fieldnames.__len__()):
			fieldname=fieldnames[i]
			if fieldname in ["ID","hostHalo","numSubStruct","npart","n_gas","n_star"]:
				fieldtype.append(np.uint64)
			elif fieldname in ["ID_mbp"]:
				fieldtype.append(np.int64)
			else:
				fieldtype.append(np.float64)
		halofile.close()
		#if desiredfields is NULL load all fields
		#but if this is passed load only those fields
		if (len(desiredfields)>0):
			lend=len(desiredfields)
			fieldindex=np.zeros(lend,dtype=int)
			desiredfieldtype=[[] for i in range(lend)]
			for i in range(lend):
				fieldindex[i]=fieldnames.index(desiredfields[i])
				desiredfieldtype[i]=fieldtype[fieldindex[i]]
			fieldtype=desiredfieldtype
			fieldnames=desiredfields
		#to store the string containing data format
		fieldtypestring=''
		for i in np.arange(fieldnames.__len__()):
			if fieldtype[i]==np.uint64: fieldtypestring+='u8,'
			elif fieldtype[i]==np.int64: fieldtypestring+='i8,'
			elif fieldtype[i]==np.float64: fieldtypestring+='f8,'

	elif (ibinary==2):
		#load hdf file
		halofile = h5py.File(filename, 'r')
		filenum=int(halofile["File_id"][0])
		numfiles=int(halofile["Num_of_files"][0])
		numhalos=np.uint64(halofile["Num_of_groups"][0])
		numtothalos=np.uint64(halofile["Total_num_of_groups"][0])
		atime=np.float(halofile.attrs["Time"]) 
		fieldnames=[str(n) for n in halofile.keys()]
		#clean of header info
		fieldnames.remove("File_id")
		fieldnames.remove("Num_of_files")
		fieldnames.remove("Num_of_groups")
		fieldnames.remove("Total_num_of_groups")
		fieldtype=[halofile[fieldname].dtype for fieldname in fieldnames]
		#if the desiredfields argument is passed only these fieds are loaded
		if (len(desiredfields)>0):
			fieldnames=desiredfields
			fieldtype=[halofile[fieldname].dtype for fieldname in fieldnames]

		#Load in the Header info so the units are known
		unitinfo = {}
		unitinfo["Mass_unit"] = halofile.attrs["Mass_unit_to_solarmass"][...]
		unitinfo["Dist_unit"] = halofile.attrs["Length_unit_to_kpc"][...]
		unitinfo["Vel_unit"] = halofile.attrs["Velocity_to_kms"][...]

		halofile.close()

	#allocate memory that will store the halo dictionary
	catalog={fieldnames[i]:np.zeros(numtothalos,dtype=fieldtype[i]) for i in range(len(fieldnames))}
	noffset=np.uint64(0)
	for ifile in range(numfiles):
		if (inompi==True): filename=basefilename+".properties"
		else: filename=basefilename+".properties"+"."+str(ifile)
		if (ibinary==0):
			halofile = open(filename, 'r')
			halofile.readline()
			numhalos=np.uint64(halofile.readline().split()[0])
			halofile.close()
			if (numhalos>0):htemp = np.loadtxt(filename,skiprows=3, usecols=fieldindex, dtype=fieldtypestring, unpack=True)
		elif(ibinary==1):
			halofile = open(filename, 'rb')
			np.fromfile(halofile,dtype=np.int32,count=2)
			numhalos=np.fromfile(halofile,dtype=np.uint64,count=2)[0]
			#halofile.seek(byteoffset);
			if (numhalos>0):htemp=np.fromfile(halofile, usecols=fieldindex, dtype=fieldtypestring, unpack=True)
			halofile.close()
		elif(ibinary==2):
			#here convert the hdf information into a numpy array
			halofile = h5py.File(filename, 'r')
			numhalos=np.uint64(halofile["Num_of_groups"][0])
			if (numhalos>0):htemp=[np.array(halofile[catvalue]) for catvalue in fieldnames]
			halofile.close()
		#numhalos=len(htemp[0])
		for i in range(len(fieldnames)):
			catvalue=fieldnames[i]
			if (numhalos>0): catalog[catvalue][noffset:noffset+numhalos]=htemp[i]
		noffset+=numhalos
	#if subhalos are written in separate files, then read them too
	if (iseparatesubfiles==1):
		for ifile in range(numfiles):
			if (inompi==True): filename=basefilename+".sublevels"+".properties"
			else: filename=basefilename+".sublevels"+".properties"+"."+str(ifile)
			if (ibinary==0):
				halofile = open(filename, 'r')
				halofile.readline()
				numhalos=np.uint64(halofile.readline().split()[0])
				halofile.close()
				if (numhalos>0):htemp = np.loadtxt(filename,skiprows=3, usecols=fieldindex, dtype=fieldtypestring, unpack=True)
			elif(ibinary==1):
				halofile = open(filename, 'rb')
				#halofile.seek(byteoffset);
				np.fromfile(halofile,dtype=np.int32,count=2)
				numhalos=np.fromfile(halofile,dtype=np.uint64,count=2)[0]
				if (numhalos>0):htemp=np.fromfile(halofile, usecols=fieldindex, dtype=fieldtypestring, unpack=True)
				halofile.close()
			elif(ibinary==2):
				halofile = h5py.File(filename, 'r')
				numhalos=np.uint64(halofile["Num_of_groups"][0])
				if (numhalos>0):htemp=[np.array(halofile[catvalue]) for catvalue in fieldnames]
				halofile.close()
			#numhalos=len(htemp[0])
			for i in range(len(fieldnames)):
				catvalue=fieldnames[i]
			if (numhalos>0): catalog[catvalue][noffset:noffset+numhalos]=htemp[i]
			noffset+=numhalos

	boxval=GadHeaderInfo["BoxSize"]*GadHeaderInfo["Scalefactor"]/GadHeaderInfo["h"]
	wdata=np.where(catalog["Xc"]<=0)
	catalog["Xc"][wdata]+=boxval
	wdata=np.where(catalog["Yc"]<=0)
	catalog["Yc"][wdata]+=boxval
	wdata=np.where(catalog["Zc"]<=0)
	catalog["Zc"][wdata]+=boxval

	wdata=np.where(catalog["Xc"]>=boxval)
	catalog["Xc"][wdata]-=boxval
	wdata=np.where(catalog["Yc"]>=boxval)
	catalog["Yc"][wdata]-=boxval
	wdata=np.where(catalog["Zc"]>=boxval)
	catalog["Zc"][wdata]-=boxval

	return catalog,numtothalos,atime,unitinfo


def GetGadgetHeader(GadFileName):
	"""
	
	Function to read the gadget2 header

	"""

	f = open(GadFileName,"rb")

	GadHeader = {}

	GadHeader["HeadLen"]       = np.fromfile(f,dtype=np.uint32,count=1)
	GadHeader["npartThisFile"] = np.fromfile(f,dtype=np.uint32,count=6)
	GadHeader["massTable"]     = np.fromfile(f,dtype=np.float64,count=6)
	GadHeader["time"]          = np.fromfile(f,dtype=np.float64,count=1)[0]
	GadHeader["redshift"]      = np.fromfile(f,dtype=np.float64,count=1)[0]
	GadHeader["flag_sfr"]      = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["flag_fb"]       = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["npartTotal"]    = np.fromfile(f,dtype=np.uint32,count=6)
	GadHeader["flag_cool"]     = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["nfiles"]        = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["boxsize"]       = np.fromfile(f,dtype=np.float64,count=1)[0]
	GadHeader["Omega0"]        = np.fromfile(f,dtype=np.float64,count=1)[0]
	GadHeader["OmegaLambda"]   = np.fromfile(f,dtype=np.float64,count=1)[0]
	GadHeader["HubbleParam"]   = np.fromfile(f,dtype=np.float64,count=1)[0]
	GadHeader["flag_age"]      = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["flag_metals"]   = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["npartTotalHW"]  = np.fromfile(f,dtype=np.uint32,count=6)
	GadHeader["flag_entropy"]         = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["flag_doubleprecision"] = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["flag_potential"]       = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["flag_fH2"]             = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["flag_tmax"]            = np.fromfile(f,dtype=np.int32,count=1)[0]
	GadHeader["flag_delaytime"]       = np.fromfile(f,dtype=np.int32,count=1)[0]

	f.close()

	return GadHeader

def GetPartSortedIndexes(comm,Rank,GadFileBasename,GadHeaderInfo,snap,Outputdir):

	start = time.time()

	if(Rank==0):
		GadSnapFilename = GadFileBasename %snap

		TotNpart = np.uint64(GadHeaderInfo["TotNpart"])

		pid = np.zeros(TotNpart,dtype=np.uint64)

		GadFileOffsets = np.zeros(GadHeaderInfo["NumFiles"]+1,dtype=np.uint64)

		ioffset = np.uint64(0)

		for i in range(GadHeaderInfo["NumFiles"]):

			GadFilename = GadSnapFilename +".%i.hdf5" %i

			GadFile = h5py.File(GadFilename,"r")

			numPartThisFile =  np.sum(GadFile["Header"].attrs["NumPart_ThisFile"],dtype=np.uint64)

			ioffset+=numPartThisFile

			pid[GadFileOffsets[i]:ioffset] = np.asarray(GadFile["PartType1"]["ParticleIDs"])

			GadFileOffsets[i+1] = ioffset

			GadFile.close()

		print("Done loading the particles in",time.time() -start)

		#Argsort the file to find where and which file they are located
		pidSortedIndexes = np.argsort(pid)

		print("Done argsorting the particles in",time.time() -start)

	else:

		GadFileOffsets = None
		pidSortedIndexes=None

	GadFileOffsets = comm.bcast(GadFileOffsets,root=0)

	return pidSortedIndexes,GadFileOffsets

def CheckForWhereWolfRestartFile(Rank, opt, apptreeFields,treeDtype):
	"""
	Function to Check for the existence of a WhereWolf restart file and read it
	"""

	if(Rank==0):
		print("Checking for restart file in",opt.outputdir)

	#Set the default values if the restart file does not exist
	newPartOffsets=None;nextPIDs=None;prevNhalo=0
	prevappendTreeData={key:np.array([],dtype=treeDtype[key]) for key in apptreeFields}

	#Varibale to keep track of what halos to track in each snapshot
	TrackData={"TrackDisp":[],"prevpos":[],"progenitor":[],"endDesc":[],"mbpSel":[],"boundSel":[],"Conc":[],"host":[],"CheckMerged":[],"TrackedNsnaps":[],"idel":[],"Rvir":[],"Mvir":[]}

	filename = opt.outputdir+"/snapshot_%03d.WW.restart.%i" %(opt.Snapshot_offset-1,Rank)

	#Check if the restart file exists
	if(os.path.isfile(filename)):

		if(opt.iverbose): print(Rank,"has found a restart file and is reading it in")

		#Open up the restart file
		WWRestartFile = h5py.File(filename,"r")

		prevNhalo = WWRestartFile.attrs["prevNhalo"][...]

		selOffsets = np.asarray(WWRestartFile["selOffsets"])[:-1].tolist()


		#Extract all the data in TrackData
		for field in TrackData.keys():

			if((field=="mbpSel") | (field=="boundSel")):

				TrackData[field] = np.split(np.asarray(WWRestartFile[field]),selOffsets)

			elif(field=="CheckMerged"):

				CheckMergedOffset = np.asarray(WWRestartFile["CheckMergedOffset"])[:-1]

				AllCheckMergedHalos = np.split(np.asarray(WWRestartFile["AllCheckMergedHalos"]),CheckMergedOffset)

				AllMergedIndicator = np.split(np.asarray(WWRestartFile["AllMergedIndicator"]),CheckMergedOffset)


				# Put the data into the check merged dictionary
				TrackData[field] = [dict(zip(iter(IDs),iter(NumSnapsWithinRvir))) for IDs, NumSnapsWithinRvir in zip(AllCheckMergedHalos,AllMergedIndicator)]

				del CheckMergedOffset

			else:

				TrackData[field] = np.asarray(WWRestartFile[field]).tolist()

		del selOffsets

		for field in apptreeFields:

			prevappendTreeData[field] = np.asarray(WWRestartFile[field])

		nextPIDs = np.asarray(WWRestartFile["nextPIDs"])

		newPartOffsets = np.asarray(WWRestartFile["newPartOffsets"])


		# if(opt.iverbose): print(Rank,"Has done loading in the restart file")

	else:

		if(opt.iverbose): print(Rank,"has not found a restart file continuing")

	return newPartOffsets, nextPIDs, prevappendTreeData, prevNhalo, TrackData


def OutputWhereWolfRestartFile(Rank, opt, snap, TrackData, newPartOffsets, nextPIDs, prevappendTreeData, prevNhalo):
	"""
	Function to create the WhereWolf restart file
	"""

	if(opt.iverbose): print(Rank,"is outputting a restart file")

	filename = opt.outputdir + "/snapshot_%03d.WW.restart.%i" %(snap+opt.Snapshot_offset,Rank)

	WWRestartFile = h5py.File(filename,"w")

	numentrys = len(TrackData["progenitor"])
	AllCheckMergedHalos = []
	AllMergedIndicator = []

	for field in TrackData.keys():


		if((field=="mbpSel") | (field=="boundSel")):

			if("selOffsets" not in WWRestartFile.keys()):

				selOffsets = np.cumsum(list(map(len,TrackData[field])))

				WWRestartFile.create_dataset("selOffsets",data=selOffsets,dtype=np.uint64)

			allSel = np.concatenate(TrackData[field])

			WWRestartFile.create_dataset(field,data=allSel,dtype=bool)


		elif(field=="CheckMerged"):

			CheckMergedOffset = np.zeros(numentrys,dtype=np.int64)

			for i,entry in enumerate(TrackData[field]):

				CheckMergedOffset[i] = len(entry) + CheckMergedOffset[i-1]

				AllCheckMergedHalos.extend(entry.keys())
				AllMergedIndicator.extend(entry.values())

			WWRestartFile.create_dataset("CheckMergedOffset",data=CheckMergedOffset,dtype=np.uint64)
			WWRestartFile.create_dataset("AllCheckMergedHalos",data=AllCheckMergedHalos,dtype=np.uint64)
			WWRestartFile.create_dataset("AllMergedIndicator",data=AllMergedIndicator,dtype=np.int16)

		else:
			WWRestartFile.create_dataset(field,data=np.asarray(TrackData[field]))

	# Also add the appendTreeData and prevNhalo to the restart file
	WWRestartFile.attrs["prevNhalo"] = prevNhalo

	for field in prevappendTreeData.keys():

		WWRestartFile.create_dataset(field,data=prevappendTreeData[field])

	WWRestartFile.create_dataset("newPartOffsets",data=newPartOffsets)
	WWRestartFile.create_dataset("nextPIDs",data=nextPIDs)





