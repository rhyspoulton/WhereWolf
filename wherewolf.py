import numpy as np 
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
	import h5py
from mpi4py import MPI
import WWio
import MPIroutines
from track import StartTrack,ContinueTrack
import time
from ui import WWOptions
import argparse

#Setup MPI
comm = MPI.COMM_WORLD
Rank = comm.Get_rank()
size = comm.Get_size()

#Get all the command line options
parser = argparse.ArgumentParser()
parser.add_argument("-c",action="store",type=argparse.FileType('r'),dest="configfile",help="Configuration file (WhereWolf.cfg)")
parser.add_argument("-s",action="store",type=int,dest="numsnaps",help="Number of snapshots")
parser.add_argument("-o",action="store",dest="outputdir",help="Output directory")
tmpOpt = parser.parse_args()

#Read the options from the config file
opt = WWOptions(tmpOpt)

#The root process tells each process what halos they are to do
if Rank==0:
	WWstatfile = open(opt.outputdir+"/WWrunstat.txt","w")
	ihalostarts, ihaloends = WWio.SetupParallelIO(opt,size)
else:
	ihalostarts=None; ihaloends=None

#Broadcast this information to each process
ihalostarts = comm.bcast(ihalostarts,root=0)
ihaloends = comm.bcast(ihaloends,root=0)

totstart = time.time()

#All the data to add for the wherewolf files
haloFields=["ID","Mass_200crit","Mass_200mean","Mass_tot","R_200crit","R_200mean","Xc","Yc","Zc","VXc","VYc","VZc","sigV","Vmax","Rmax","hostHaloID","npart","Num_of_files","Group_Size","Particle_IDs","Particle_IDs_unbound","Parent_halo_ID"]
updatetreeFields=["ID","Descendants"]
apptreeFields=["ID","NumDesc","Ranks","Descendants","endDesc"]
treeDtype={"ID":"uint64","NumDesc":"int32","Ranks":"int32","Descendants":"uint64","endDesc":"uint64"}
WWstatkeys = ["TotStartTracked","NoStart","Start","TotContTrack","PartLimitStart","Merged","notMerged","Match","MatchStart","MatchCore","MatchStartCore","Mixed","ConnectPartLimit","Connect","contNSnap"]

#Setup the stat file
if(Rank==0):
	WWstatfile.write("# "+" ".join(WWstatkeys) + "\n")

#Varibale to keep track of what halos to track in each snapshot
TrackData={"TrackDisp":[],"prevpos":[],"progenitor":[],"endDesc":[],"mbpSel":[],"boundSel":[],"Conc":[],"host":[],"CheckMerged":[],"TrackedNsnaps":[],"idel":[],"Rvir":[],"Mvir":[]}

newPartOffsets=None;nextPIDs=None;prevappendTreeData=None;prevNhalo=None

#Boolean array to keep track if halos have been tracked in the snapshot
TrackFlag = np.zeros(opt.numsnaps,dtype=bool)

fsnap = opt.numsnaps-1

for isnap in range(opt.numsnaps):

	WWstat ={key:0 for key in WWstatkeys}

	if(Rank==0):
		print("Doing snap",isnap)

	#Check if any process has any halos
	if(np.sum(ihaloends[:,isnap])==0):
		if(Rank==0):
			print("There are no halos present at this snapshot")
		continue


	start = time.time()

	ihalostart = ihalostarts[Rank][isnap]
	ihaloend = ihaloends[Rank][isnap]
		

	numhalos = ihaloend - ihalostart


	#Extract the header info from the gadget snapshot
	if Rank==0:
		GadHeaderInfo = WWio.GetGadFileInfo(opt.GadFileList[isnap])
	else:
		GadHeaderInfo=None

	#Broadcast to the other processes
	GadHeaderInfo = comm.bcast(GadHeaderInfo,root=0)

	#Reset the number of halos tracked
	nTracked = 0

	#Read the VELOCIraptor property file and the treefrog tree
	snapdata, totnumhalos, atimex  = WWio.ReadPropertyFile(opt.VELFileList[isnap],GadHeaderInfo,ibinary=2,desiredfields = ["ID","Mass_200crit","R_200crit","Xc","Yc","Zc","VXc","VYc","VZc","hostHaloID","cNFW","npart"])
	treedata = WWio.ReadVELOCIraptorTreeDescendant(opt.TreeFileList[isnap])

	#Open up the VELOCIraptor files to read the halo particle info
	filenumhalos,VELnumfiles,pfiles,upfiles,grpfiles = WWio.OpenVELOCIraptorFiles(opt.VELFileList[isnap])


	if((isnap<fsnap) & (numhalos>0)):

		# npart = WWio.ReadVELOIraptorCatalogueNpartSplit(VELFilename,ihalostart[Rank][snapindex],ihaloend[Rank][snapindex],VELfilenumhalos)
		# tree = WWio.ReadVELOCIraptorTreeDescendant(TreeFilename,ihalostart[Rank][snapindex],ihaloend[Rank][snapindex])

		# Select where the halo has merged with more than 50 particles
		TrackDispSel = (treedata["Rank"][ihalostart:ihaloend]>0)  

		#Find where there are gaps in the tree
		TrackFillSel = (treedata["Rank"][ihalostart:ihaloend]==0) & (((treedata["Descen"][ihalostart:ihaloend]/opt.Temporal_haloidval).astype(int) - isnap+opt.Snapshot_offset)>1)

		#See if the descendant has a merit below the hard merit limit
		TrackMerit = (treedata["Merits"][ihalostart:ihaloend]<=0.2) & (treedata["Rank"][ihalostart:ihaloend]>-1)

		#Find where the rank is greater than zero
		trackIndx = np.where((TrackDispSel | TrackFillSel | TrackMerit) & (snapdata["npart"][ihalostart:ihaloend]>50))[0]

		trackMergeDesc = treedata["Descen"][ihalostart:ihaloend][trackIndx]

		trackDispFlag = treedata["Rank"][ihalostart:ihaloend][trackIndx]>0

		tracknpart= snapdata["npart"][ihalostart:ihaloend][trackIndx]

		trackIndx += ihalostart

	else:

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
		if(Rank==0):
			print("All the processes have found nothing to track at this snapshot")
		WWio.CloseVELOCIraptorFiles(opt.VELFileList[isnap],VELnumfiles,pfiles,upfiles,grpfiles)
		continue

	if(Rank==0):

		print("Attempting to track",NtotTrack,"halos and continuing to track",NtrackNextSnap,"halos")
		

	if((ntrack>0) | (ntrackNextSnap>0)):
		allpid,allpartpos,allpartvel,allPartOffsets = WWio.GetParticleData(Rank,size,opt,isnap,trackIndx,tracknpart,GadHeaderInfo,VELnumfiles,filenumhalos,pfiles,upfiles,grpfiles,newPartOffsets,nextPIDs)


	if(Rank==0):

		print("Done loading particles in",time.time()-start)

	startPartOffsets=None;newPartOffsets=None;nextPIDs=None;pidOffset=0
		
	
	
	#Track the halos one snapshot forwad if they need it, this is only done once there has been found that there are halos to be tracked in the next snapshot
	if(NtrackNextSnap>0):
		#Create a dataset to store all the halo and tree data
		appendHaloData={key:[] for key in haloFields}
		appendTreeData={key:[] for key in apptreeFields}
		prevupdateTreeData={key:[] for key in updatetreeFields}
		appendHaloData["Num_of_groups"]=np.array([0])
		appendHaloData["File_id"]=np.array([0])
		appendHaloData["Offset"]=[0]
		appendHaloData["Offset_unbound"]=[0]
		appendHaloData["Particle_IDs_unbound"]=np.array([])

		nTracked = len(TrackData["progenitor"]) 

		#If thre are halos to track then lers track them into the next snapshot
		if(nTracked>0):
			newPartOffsets,contPIDs = ContinueTrack(opt,isnap,fsnap,TrackData,allpid,allpartpos,allpartvel,allPartOffsets,snapdata,treedata,filenumhalos,pfiles,upfiles,grpfiles,GadHeaderInfo,appendHaloData,appendTreeData,prevappendTreeData,prevupdateTreeData,prevNhalo,WWstat)
			pidOffset=len(contPIDs)

		#Now done Tracking lets turn the output data into arrays for easy indexing
		for key in apptreeFields:	appendTreeData[key] = np.asarray(appendTreeData[key],dtype=treeDtype[key])
		for key in updatetreeFields:	prevupdateTreeData[key] = np.asarray(prevupdateTreeData[key],dtype=treeDtype[key])
		for key in haloFields: appendHaloData[key] = np.asarray(appendHaloData[key])


		prevNhalo=len(snapdata["ID"])

		#If the number of threads is > 1 then need to update IDs and gather the TreeData onto the root thread
		if(size>1):

			appendHaloData,prevupdateTreeData,appendTreeData,prevappendTreeData = MPIroutines.UpdateIDsoffsets(comm,Rank,size,appendHaloData,prevupdateTreeData,appendTreeData,prevappendTreeData,prevNhalo)	

			prevappendTreeData,prevupdateTreeData=MPIroutines.GatheroutputTreeData(comm,Rank,size,prevappendTreeData,prevupdateTreeData,treeDtype)


		#Need to close the VELOCIraptor files before outputing the data
		WWio.CloseVELOCIraptorFiles(opt.VELFileList[isnap],VELnumfiles,pfiles,upfiles,grpfiles)

		#Find the total amount of halos to be appended
		Nappend = len(appendHaloData["ID"])
		TotNappend = int(0)
		TotNappend = comm.allreduce(Nappend,MPI.SUM)

		#Add WW VELOCIraptor file per process while updating the VELOCIraptor files
		# WWio.AddWhereWolfFileParallel(comm,Rank,size,opt.VELFileList[isnap],appendHaloData,Nappend,TotNappend)

		# Set in the flag that a treefile has been created for a snapshot before
		TrackFlag[isnap-1] = True

		# If the rootprocess then write to the treedata
		if(Rank==0):

			print("Total num halos:",TotNappend,prevNhalo,TotNappend + prevNhalo)
			# The file is for the previous snapshot
			WWio.OutputWhereWolfTreeData(opt,isnap-1,prevappendTreeData,prevupdateTreeData)

		prevappendTreeData=appendTreeData
		

		del appendHaloData
		del prevupdateTreeData

	else:
		# Close all the files
		WWio.CloseVELOCIraptorFiles(opt.VELFileList[isnap],VELnumfiles,pfiles,upfiles,grpfiles)


	#Try to find halos to track if not at the last snapshot
	if((isnap<fsnap) & (ntrack>0)):
		startPartOffsets,startPIDs  = StartTrack(opt,trackIndx,trackMergeDesc,trackDispFlag,allpid,allpartpos,allpartvel,allPartOffsets[nTracked:],GadHeaderInfo,snapdata,treedata,TrackData,pidOffset,WWstat)

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

	if(Rank==0):
		print(isnap,"Done in",time.time()-start)

		for field in WWstatkeys:
			WWstatfile.write("%i "%ALLWWstat[field])
		WWstatfile.write("\n")

#Check if anything has been tracked
if(any(TrackFlag)):

	#Gather the final tree data
	appendTreeData,prevupdateTreeData=MPIroutines.GatheroutputTreeData(comm,Rank,size,appendTreeData,None,treeDtype)

	#If the root process then generate the final treefile and filelists
	if(Rank==0):
		WWstatfile.close()

		#Output the final tree file
		WWio.OutputWhereWolfTreeData(opt,isnap,appendTreeData,None)
		TrackFlag[isnap] = True


		#Create the file list in the output directory
		treefilelist = open(opt.outputdir+"treesnaplist.txt","w")
		for isnap in range(opt.numsnaps):

			#Write out the WW treefile name if it has been tracked or the original tree name if not
			if(TrackFlag[isnap]):
				treefilelist.write( opt.outputdir+"/snapshot_%03d.VELOCIraptor.WW.tree\n" %(isnap+opt.Snapshot_offset))
			else:
				treefilelist.write(opt.TreeFileList[isnap] + "\n")
		treefilelist.close()

		# snaplist = open(OutputDir+"snaplist.txt","w")
		# for snap in range(isnap,fsnap+1):
		# 	snaplist.write(VELFileBasename %(snap) +"\n")
		# snaplist.close()

		print("Tracking done in",time.time() - totstart)
		print("With WhereWolf halos, particles are no longer exclusive to a single halo")
