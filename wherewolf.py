
import sys
sys.path.append(sys.argv[0].replace("wherewolf.py",""))
sys.path.append(sys.argv[0].replace("wherewolf.py","")+"VELOCIraptor_Python_Tools")
import numpy as np 
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	import h5py
from mpi4py import MPI
import WWio
import MPIroutines
from track import StartTrack,ContinueTrack
from utils import CheckHaloHasProgen
import time
from ui import WWOptions
import argparse
import os
import velociraptor_python_tools as VPT


#Setup MPI
comm = MPI.COMM_WORLD
Rank = comm.Get_rank()
size = comm.Get_size()

#Get all the command line options
parser = argparse.ArgumentParser()
parser.add_argument("-c",action="store",type=argparse.FileType('r'),dest="configfile",help="Configuration file (wherewolf.cfg)",required=True)
parser.add_argument("-n",action="store",type=int,dest="numsnaps",help="Number of snapshots",required=True)
parser.add_argument("-o",action="store",dest="outputdir",help="Output directory",required=True)
parser.add_argument("-s",action="store",default=0,type=int,dest="Snapshot_offset",help="The offset if not starting at snapshot 0 in the simulation, default: 0")
tmpOpt = parser.parse_args()

#All the data to add for the wherewolf files
haloFields=["ID","Mass_200crit","Mass_200mean","Mass_tot","R_200crit","R_200mean","Xc","Yc","Zc","VXc","VYc","VZc","sigV","Vmax","Rmax","hostHaloID","npart","Num_of_files","Group_Size","Particle_IDs","Particle_IDs_unbound","Parent_halo_ID"]
updatetreeFields=["ID","Descendants","Merits"]
apptreeFields=["ID","NumDesc","Ranks","Descendants","Merits"]
treeDtype={"ID":"uint64","NumDesc":"int32","Ranks":"int32","Descendants":"uint64","Merits":"float32"}
WWstatkeys = ["TotStartTracked","StartTrackDisp","StartFill","NoStart","Start","TotContTrack","PartLimitStart","Merged","MergedMBP","Matched","notMerged","Mixed","ConnectPartLimit","Connect","ConnectMBP","contNSnap"]


#Read the options from the config file
opt = WWOptions(comm,Rank,tmpOpt)

#Check if a WhereWolf restart file exists and read it in if it does
newPartOffsets, nextPIDs, prevappendTreeData, prevNhalo, TrackData = WWio.CheckForWhereWolfRestartFile(Rank, opt, apptreeFields,treeDtype)

RestartFlag = True if(len(TrackData["progenitor"])>0) else False

#The root process tells each process what halos they are to do
if Rank==0:

	#If starting at a nonzero snapshot and the runstat file exists then append the data to it
	if((opt.Snapshot_offset>0) & (os.path.exists(opt.outputdir+"/WWrunstat.txt"))):
		WWstatfile = open(opt.outputdir+"/WWrunstat.txt","a")
	else:
		WWstatfile = open(opt.outputdir+"/WWrunstat.txt","w")
		WWstatfile.write("# "+" ".join(WWstatkeys) + "\n")
	ihalostarts, ihaloends, allnumhalos = WWio.SetupParallelIO(comm,opt,size)
else:
	ihalostarts=None; ihaloends=None;  allnumhalos=None;

#Broadcast this information to each process
ihalostarts = comm.bcast(ihalostarts,root=0)
ihaloends = comm.bcast(ihaloends,root=0)
allnumhalos = comm.bcast(allnumhalos,root=0)

#List of arrays to mark if the halo has a progenitor
progenBool = [[] for i in range(opt.numsnaps)]

totstart = time.time()

#Boolean array to keep track if halos have been tracked in the snapshot
TrackFlag = np.zeros(opt.numsnaps,dtype=bool)

TotNappend = int(0)
prevTotNappend = int(0)


#Create a dataset to store all the halo and tree data
appendHaloData={key:[] for key in haloFields}
appendTreeData={key:[] for key in apptreeFields}
prevupdateTreeData={key:[] for key in updatetreeFields}
appendHaloData["Num_of_groups"]=np.array([0])
appendHaloData["Num_of_files"]=np.int32(size)
appendHaloData["File_id"]=np.int32(Rank)
appendHaloData["Offset"]=[0]
appendHaloData["Offset_unbound"]=[0]
appendHaloData["Particle_IDs_unbound"]=np.array([])

for isnap in range(opt.numsnaps):

	snap = isnap + opt.Snapshot_offset

	WWstat ={key:0 for key in WWstatkeys}

	if(Rank==0):
		print("Doing snap",snap)

	#Check if any process has any halos
	if(np.sum(ihaloends[:,isnap])==0):
		if(Rank==0):
			print("There are no halos present at this snapshot")
		continue


	starttot = time.time()

	ihalostart = ihalostarts[Rank][isnap]
	ihaloend = ihaloends[Rank][isnap]
		

	numhalos = ihaloend - ihalostart


	#Extract the header info from the gadget snapshot
	if Rank==0:
		GadHeaderInfo = WWio.GetGadFileInfo(comm,opt.GadFileList[snap])
	else:
		GadHeaderInfo=None

	#Broadcast to the other processes
	GadHeaderInfo = comm.bcast(GadHeaderInfo,root=0)

	#Reset the number of halos tracked
	nTracked = 0

	if(Rank==0):
		#Read the VELOCIraptor property file and the treefrog tree
		snapdata, totnumhalos  = VPT.ReadPropertyFile(opt.VELFileList[snap],ibinary=2,desiredfields = ["ID","Mass_200crit","R_200crit","Xc","Yc","Zc","VXc","VYc","VZc","hostHaloID","cNFW","npart"])
		WWio.AdjustforPeriod(snapdata)
		treeOpt,treedata = WWio.ReadVELOCIraptorTreeDescendant(comm,opt.TreeFileList[snap])
	else:
		snapdata = None; totnumhalos= None
		treeOpt=None;treedata =None


	snapdata = comm.bcast(snapdata,root=0)
	totnumhalos = comm.bcast(totnumhalos,root=0)
	treedata = comm.bcast(treedata,root=0)
	treeOpt = comm.bcast(treeOpt,root=0)

	#Populate the progenBool to mark if halos have a progenitor
	if("Descen" in treedata.keys()):
		#Find the minimum of the number of steps needed to look forward, this is only needs to be done if the halos have descendant
		istep=min(treeOpt["Number_of_steps"],opt.numsnaps-1 - isnap)

		#Set the if a halo has progenitor and communicate this to all processes
		updateindexes=CheckHaloHasProgen(opt,treeOpt,isnap,istep,treedata,numhalos,allnumhalos,ihalostart,ihaloend,progenBool)
		MPIroutines.CommunicateProgenBool(comm ,Rank,size,isnap,istep,updateindexes,progenBool)


	if(Rank==0):
		print("Done loading halo and tree data in",time.time()-starttot,"now finding halos to track")

	#Open up the VELOCIraptor files to read the halo particle info
	filenumhalos,VELnumfiles,pfiles,upfiles,grpfiles = WWio.OpenVELOCIraptorFiles(opt.VELFileList[snap])

	if(("Descen" in treedata.keys()) & (numhalos>0)):

		#Check if the  VELOCIraptor and TreeFrog catalogues sizes match up
		if(snapdata["ID"].size!=treedata["Descen"].size):
			if(Rank==0):
				print("The number of halos in the VELOCIraptor catalogue (%i) does not match the number in the TreeFrog tree(%i), please check they are the correct files." %(snapdata["ID"].size,treedata["Descen"].size))
				print("Terminating")
				comm.Abort()
			else:
				comm.barrier()

		# npart = WWio.ReadVELOIraptorCatalogueNpartSplit(VELFilename,ihalostart[Rank][snapindex],ihaloend[Rank][snapindex],VELfilenumhalos)
		# tree = WWio.ReadVELOCIraptorTreeDescendant(TreeFilename,ihalostart[Rank][snapindex],ihaloend[Rank][snapindex])

		# Select where the halo has merged with more than 50 particles
		TrackDispSel = (treedata["Rank"][ihalostart:ihaloend]>0)  | (treedata["NumDesc"][ihalostart:ihaloend]==0)

		#Find where there are gaps in the tree
		TrackFillSel = (treedata["Rank"][ihalostart:ihaloend]==0) & (((treedata["Descen"][ihalostart:ihaloend]/opt.Temporal_haloidval).astype(int) - snap)>1)

		trackIndx = np.where((TrackDispSel | TrackFillSel) & (snapdata["npart"][ihalostart:ihaloend]>50))[0]

		trackMergeDesc = treedata["Descen"][ihalostart:ihaloend][trackIndx]

		trackDispFlag = (treedata["Rank"][ihalostart:ihaloend][trackIndx]>0) | (treedata["NumDesc"][ihalostart:ihaloend][trackIndx]==0)

		tracknpart= snapdata["npart"][ihalostart:ihaloend][trackIndx]

		trackIndx += ihalostart

	else:

		tracknpart = np.array([])

		trackIndx = np.array([])

	# Get the total amount of halos to track from all processes
	ntrack = np.int64(len(trackIndx))
	NtotTrack = np.int64(0)
	NtotTrack = comm.allreduce(ntrack,MPI.SUM)	

	#Find if there is anything that is tracked from all the processes
	ntrackNextSnap = len(TrackData["progenitor"])
	NtrackNextSnap = int(0)
	NtrackNextSnap = comm.allreduce(ntrackNextSnap,MPI.SUM)	

	WWstat["TotStartTracked"]=ntrack
	WWstat["TotContTrack"]=ntrackNextSnap

	#See if any of the processors has anything to track
	if((NtotTrack==0) & (NtrackNextSnap==0)):

		#Lets check and see if there is treedata to gather
		if(prevTotNappend>0):
			prevappendTreeData,prevupdateTreeData=MPIroutines.GatheroutputTreeData(comm,Rank,size,prevappendTreeData,None,treeDtype)

		if(Rank==0):
			print("All the processes have found nothing to track at this snapshot")

			#If there was treedata to gather lets output it before moving onto the next snapshot
			if(prevTotNappend>0):
				# Set in the flag that a treefile has been created for a snapshot before
				TrackFlag[isnap-1] = True

				WWio.OutputWhereWolfTreeData(opt,snap-1,prevappendTreeData,None)
				prevTotNappend = 0
				prevappendTreeData = {key:np.array([]) for key in apptreeFields}

		WWio.CloseVELOCIraptorFiles(opt.VELFileList[snap],VELnumfiles,pfiles,upfiles,grpfiles)
		continue

	if(Rank==0):
		print("Attempting to track",NtotTrack,"halos and continuing to track",NtrackNextSnap,"halos")
		start = time.time()

	if((ntrack>0) | (ntrackNextSnap>0)):
		allpid,allpartpos,allpartvel,allPartOffsets = WWio.GetParticleData(comm,Rank,size,opt,isnap,trackIndx,tracknpart,GadHeaderInfo,VELnumfiles,filenumhalos,pfiles,upfiles,grpfiles,newPartOffsets,nextPIDs)


	if(Rank==0):

		print("Done loading particles in",time.time()-start)

	startPartOffsets=None;newPartOffsets=None;nextPIDs=None;pidOffset=0

	
	#Track the halos one snapshot forwad if they need it, this is only done once there has been found that there are halos to be tracked in the next snapshot
	if(NtrackNextSnap>0):

		nTracked = len(TrackData["progenitor"]) 
		#If thre are halos to track then lers track them into the next snapshot
		if(nTracked>0):
			newPartOffsets,contPIDs = ContinueTrack(opt,isnap,TrackData,allpid,allpartpos,allpartvel,allPartOffsets,snapdata,treedata,progenBool[isnap],filenumhalos,pfiles,upfiles,grpfiles,GadHeaderInfo,appendHaloData,appendTreeData,prevappendTreeData,prevupdateTreeData,prevNhalo,WWstat,treeOpt)
			pidOffset=len(contPIDs)

		#Now done Tracking lets turn the output data into arrays for easy indexing
		for key in apptreeFields:	appendTreeData[key] = np.asarray(appendTreeData[key],dtype=treeDtype[key])
		for key in apptreeFields:	prevappendTreeData[key] = np.asarray(prevappendTreeData[key],dtype=treeDtype[key])
		for key in updatetreeFields:	prevupdateTreeData[key] = np.asarray(prevupdateTreeData[key],dtype=treeDtype[key])
		for key in haloFields: appendHaloData[key] = np.asarray(appendHaloData[key])

		prevNhalo=len(snapdata["ID"])

		#If the number of threads is > 1 then need to update IDs and gather the TreeData onto the root thread
		if(size>1):

			appendHaloData,prevupdateTreeData,appendTreeData,prevappendTreeData = MPIroutines.UpdateIDsoffsets(comm,Rank,size,snap,appendHaloData,prevupdateTreeData,appendTreeData,prevappendTreeData,prevNhalo)

			prevappendTreeData,prevupdateTreeData=MPIroutines.GatheroutputTreeData(comm,Rank,size,prevappendTreeData,prevupdateTreeData,treeDtype)

		#Need to close the VELOCIraptor files before outputing the data
		WWio.CloseVELOCIraptorFiles(opt.VELFileList[snap],VELnumfiles,pfiles,upfiles,grpfiles)

		#Find the total amount of halos to be appended
		Nappend = len(appendHaloData["ID"])
		TotNappend = int(0)
		TotNappend = comm.allreduce(Nappend,MPI.SUM)

		#Only need to update VELOCIraptor files if there are halos to append
		if(TotNappend>0):

			if(Rank==0):
				print("Total num halos:",TotNappend,prevNhalo,TotNappend + prevNhalo)

			#Add WW VELOCIraptor file per process while updating the VELOCIraptor files
			WWio.AddWhereWolfFileParallel(comm,Rank,size,opt,snap,appendHaloData,Nappend,TotNappend,snapdata["UnitInfo"])

		#Reset the halodata once it has been outputted
		appendHaloData={key:[] for key in haloFields}
		appendHaloData["Num_of_groups"]=np.array([0])
		appendHaloData["File_id"]=np.array([0])
		appendHaloData["Offset"]=[0]
		appendHaloData["Offset_unbound"]=[0]
		appendHaloData["Particle_IDs_unbound"]=np.array([])

		# Set in the flag that a treefile has been created for a snapshot before
		TrackFlag[isnap-1] = True

		#Done with the merits at this snapshot so the array can be deallocated
		progenBool[isnap] = []

	else:
		# Close all the files
		WWio.CloseVELOCIraptorFiles(opt.VELFileList[snap],VELnumfiles,pfiles,upfiles,grpfiles)

	# If the rootprocess then write to the treedata
	if((Rank==0) & ((prevTotNappend>0) | (len(prevupdateTreeData["ID"])>0))):

		# The file is for the previous snapshot
		WWio.OutputWhereWolfTreeData(opt,snap-1,prevappendTreeData,prevupdateTreeData)

	prevappendTreeData=appendTreeData

	prevTotNappend = TotNappend

	#Reset the treedata once it has been outputted
	appendTreeData={key:[] for key in apptreeFields}
	prevupdateTreeData={key:[] for key in updatetreeFields}
	

	#Try to find halos to track if not at the last snapshot
	if(("Descen" in treedata.keys()) & (ntrack>0)):
		startPartOffsets,startPIDs  = StartTrack(opt,snap,trackIndx,trackMergeDesc,trackDispFlag,allpid,allpartpos,allpartvel,allPartOffsets[nTracked:],GadHeaderInfo,snapdata,treedata,TrackData,pidOffset,WWstat)

	#Update the nextPIDS and the newPartOffsets
	if((newPartOffsets is not None) & (startPartOffsets is not None)):
		newPartOffsets.extend(startPartOffsets)
		nextPIDs = np.zeros(len(contPIDs)+len(startPIDs),dtype=np.uint64)
		nextPIDs[:pidOffset]=contPIDs
		nextPIDs[pidOffset:]=startPIDs
		del contPIDs
		del startPIDs
	elif(startPartOffsets is not None):
		newPartOffsets=startPartOffsets
		nextPIDs = startPIDs
	elif(newPartOffsets is not None):
		nextPIDs = contPIDs

	ALLWWstat = {}

	for field in WWstat.keys():
		data = int(WWstat[field])
		ALLWWstat[field] = 0
		ALLWWstat[field] = comm.reduce(data,MPI.SUM,root=0)

	#Wait for all processes to finish this snapshot before moving onto the next one
	comm.barrier()

	#Update the WWstat file with the statistics from  this snapshot
	if(Rank==0):
		print(snap,"Done in",time.time()-starttot)
		WWstatfile.write("%i " %snap)
		for field in WWstatkeys:
			WWstatfile.write("%i "%ALLWWstat[field])
		WWstatfile.write("\n")


if(len(TrackData["progenitor"])>0):
	WWio.OutputWhereWolfRestartFile(Rank, opt, isnap, TrackData, newPartOffsets, nextPIDs, prevappendTreeData, prevNhalo)

#Close the WWstat file
if(Rank==0): WWstatfile.close()

#Check if anything has been tracked
if(any(TrackFlag)):

	#Gather the final tree data
	appendTreeData,prevupdateTreeData=MPIroutines.GatheroutputTreeData(comm,Rank,size,appendTreeData,prevupdateTreeData,treeDtype)

	#If the root process then generate the final treefile and filelists
	if(Rank==0):

		#Output the final tree file
		WWio.OutputWhereWolfTreeData(opt,snap,appendTreeData,prevupdateTreeData)
		TrackFlag[isnap] = True


		#Create the file list in the output directory
		if((opt.Snapshot_offset>0) & (os.path.exists(opt.outputdir+"/treesnaplist.txt"))):
			treefilelist = open(opt.outputdir+"/treesnaplist.txt","a")
		else:
			treefilelist = open(opt.outputdir+"/treesnaplist.txt","w")
		for isnap in range(opt.numsnaps):

			#Write out the WW treefile name if it has been tracked or the original tree name if not
			if(TrackFlag[isnap]):
				treefilelist.write( opt.outputdir+"/snapshot_%03d.VELOCIraptor.WW\n" %(isnap+opt.Snapshot_offset))
			else:
				treefilelist.write(opt.TreeFileList[isnap+opt.Snapshot_offset] + "\n")
		treefilelist.close()

		if((opt.Snapshot_offset>0) & (os.path.exists(opt.outputdir+"/WWsnaplist.txt"))):
			snaplist = open(opt.outputdir+"/WWsnaplist.txt","a")
		else:
			snaplist = open(opt.outputdir+"/WWsnaplist.txt","w")

		for isnap in range(opt.numsnaps):
			snaplist.write(opt.outputdir+"snapshot_%03d.VELOCIraptor.WW" %(isnap+opt.Snapshot_offset) +"\n")
		snaplist.close()

		print("Tracking done in",time.time() - totstart)
		print("With WhereWolf halos, particles are no longer exclusive to a single halo")
