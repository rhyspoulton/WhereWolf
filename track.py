import numpy as np
import time
import mmap
import cosFuncs as cf
import WWio
from scipy.spatial import cKDTree
from utils import CalculateMerit,MergeHalo


G = 43.0211349

def StartTrack(opt,snap,trackIndx,trackMergeDesc,trackDispFlag,allpid,allpartpos,allpartvel,partOffsets,GadHeaderInfo,snapdata,treedata,TrackData,pidOffset,WWstat,unitinfo):

	#Find the physical boxsize
	boxsize = GadHeaderInfo["BoxSize"]*GadHeaderInfo["Scalefactor"]/GadHeaderInfo["h"] * unitinfo["Dist_unit"]/1000.0 # Mpc

	keepPIDS = np.zeros(len(allpid),dtype = bool)
	newPartOffset=pidOffset
	newPartOffsets=[]

	for i,Indx in enumerate(trackIndx):		
		#Get the particle offset 
		partOffset=partOffsets[i]

		#Extract all the required properties
		halopos=np.asarray([snapdata["Xc"][Indx],snapdata["Yc"][Indx],snapdata["Zc"][Indx]])
		halovel=np.asarray([snapdata["VXc"][Indx],snapdata["VYc"][Indx],snapdata["VZc"][Indx]])
		conc=snapdata["cNFW"][Indx]
		Mass_200crit=snapdata["Mass_200crit"][Indx]
		R_200crit=snapdata["R_200crit"][Indx]
		npart =snapdata["npart"][Indx]


		#Find the id and index of the host
		if(snapdata["hostHaloID"][Indx]>-1):

			Hostindx=int(snapdata["hostHaloID"][Indx]%opt.Temporal_haloidval-1)
			tmpHostHead=treedata["Descen"][Hostindx]

			#Lets check this host halo is in thr next snapshot
			hostHeadSnap = int(tmpHostHead/opt.Temporal_haloidval)
			if(hostHeadSnap==snap+1):
				hostHead=tmpHostHead
			else:
				hostHead=-1

		else:
			hostHead=-1

		partpos = allpartpos[partOffset:partOffset+npart] * unitinfo["Dist_unit"]/1000.0 # Mpc
		partvel = allpartvel[partOffset:partOffset+npart] * unitinfo["Vel_unit"]

		#Correct for the periodicity
		for j in range(3):
			if(np.ptp(partpos[:,j])>0.5*boxsize):

				#Find which boundary the halo lies
				boundary = -boxsize if(halopos[j]<0.5*boxsize) else boxsize

				# prevpos = np.mean(partpos[:,0][TrackData["boundSel"][i]])
				sel = abs(partpos[:,j]-halopos[j])>0.5*boxsize

				partpos[:,j][sel] = partpos[:,j][sel] + boundary

		#Get the radial posistions and velocities relative to the halo's
		r=np.sqrt((partpos[:,0]-halopos[0])**2 + (partpos[:,1]-halopos[1])**2 + (partpos[:,2]-halopos[2])**2)
		vr=np.sqrt((partvel[:,0]-halovel[0])**2 + (partvel[:,1]-halovel[1])**2 + (partvel[:,2]-halovel[2])**2)

		#Calculate the circular velocities for the particles in the halo
		Vesc_calc=cf.cosNFWvesc(r, Mass_200crit, conc,z = GadHeaderInfo["Redshift"],f=1,Munit=unitinfo["Mass_unit"],Dist="Ang")

		#Iterate on posistion until less than 100 particles or tracking less than 10% of the halos paricles
		j=0

		mbpsel = (vr<Vesc_calc*0.8/(2**j)) & ((r/R_200crit)<0.8/(2**j))
		while(True):
			mbpnpart=np.sum(mbpsel)
			if((mbpnpart<100) | ((mbpnpart/npart)<0.1)):
				if((j>1) & (mbpnpart<10)):
					# print("Iterated to core with less than 10 mpb",np.sum(mbpsel),"returning to the previous selection of",np.sum(tmpMBPsel),"from a",npart,"size halo")
					mbpsel = tmpMBPsel
				break

			tmpMBPsel = mbpsel
			mbpsel=(vr/Vesc_calc<0.8/(2**j) ) & ((r/R_200crit)<0.8/(2**j))
			j+=1

		#Check to see if the halo can still be tracked 
		if(mbpnpart<=5):
			WWstat["NoStart"]+=1
			# print(i,"There are less than 10 highly bound particles to track posistion terminating the tracking")
			continue
		WWstat["Start"]+=1 


		#If it can be tracked then add it to the tracked data to be tracked
		TrackData["progenitor"].append(snapdata["ID"][Indx])

		#Lets see if the halo is tracked until dispersed or filling in of a gap
		TrackData["prevpos"].append(halopos)
		TrackData["TrackDisp"].append(trackDispFlag[i])
		TrackData["endDesc"].append(trackMergeDesc[i])
		TrackData["mbpSel"].append(mbpsel)
		TrackData["boundSel"].append(np.ones(npart,dtype=bool))
		TrackData["Conc"].append(conc)
		TrackData["host"].append(hostHead)
		TrackData["CheckMerged"].append({})
		TrackData["TrackedNsnaps"].append(0)
		TrackData["idel"].append(0)
		TrackData["Rvir"].append(R_200crit)
		TrackData["Mvir"].append(Mass_200crit)

		#Update the keepPIDS array so the PID is kept
		keepPIDS[partOffset:partOffset+npart]=True

		#Update the offsets
		newPartOffsets.append(np.uint64(newPartOffset))

		#Add to the offset for the next halo
		newPartOffset+=npart

	startpids = allpid[keepPIDS]

	return newPartOffsets,startpids


def ContinueTrack(opt,snap,TrackData,allpid,allpartpos,allpartvel,partOffsets,snapdata,treedata,progenBool,filenumhalos,pfiles,upfiles,grpfiles,GadHeaderInfo,appendHaloData,appendTreeData,prevappendTreeData,prevupdateTreeData,prevNhalo,WWstat,treeOpt,unitinfo):

	#Find the physical boxsize
	boxsize = GadHeaderInfo["BoxSize"]*GadHeaderInfo["Scalefactor"]/GadHeaderInfo["h"] * unitinfo["Dist_unit"]/1000.0 # Mpc

	#Find rho crit and use that to find R200crit and M200crit
	rhocrit = cf.cosgrowRhoCrit(z = GadHeaderInfo["Redshift"])
	rhomean = cf.cosgrowRhoMean(z = GadHeaderInfo["Redshift"])

	keepPIDS = np.zeros(len(allpid),dtype = bool)
	newPartOffset=0
	newPartOffsets=[]

	imerittype=6

	#Create a spatial kdtree to find the nearest centrals neighbours to the halo
	pos=np.array([snapdata["Xc"],snapdata["Yc"],snapdata["Zc"]])
	pos_tree=cKDTree(pos.T,boxsize=boxsize)

	medianPartSpread = np.zeros(len(TrackData["progenitor"]))

	for i in range(len(TrackData["progenitor"])):

		partOffset = partOffsets[i]
		totnpart = np.uint64(TrackData["boundSel"][i].size)

		partpos = allpartpos[partOffset:partOffset+totnpart] * unitinfo["Dist_unit"]/1000.0 # Mpc
		partvel = allpartvel[partOffset:partOffset+totnpart] * unitinfo["Vel_unit"]
		partIDs = allpid[partOffset:partOffset+totnpart]

		#Correct for the periodicity
		for j in range(3):
			if(np.ptp(partpos[:,j][TrackData["boundSel"][i]])>0.5*boxsize):
				

				#Find which boundary the core lies
				boundary = -boxsize if(TrackData["prevpos"][i][j]<0.5*boxsize) else boxsize

				# prevpos = np.mean(partpos[:,0][TrackData["boundSel"][i]])
				sel = (abs(partpos[:,j]-TrackData["prevpos"][i][j])>0.5*boxsize) & (TrackData["boundSel"][i])

				partpos[:,j][sel] = partpos[:,j][sel] + boundary

		#Find the average position and velocities of the most bound particles
		mbppos=partpos[TrackData["mbpSel"][i]]
		meanpos=np.mean(mbppos,axis= 0)
		mbpvel=partvel[TrackData["mbpSel"][i]]
		meanvel=np.mean(mbpvel,axis=0)

		#Update the prevoius posistion of the halo
		TrackData["prevpos"][i] = meanpos

		#Store the previous bound selection
		prevboundSel = TrackData["boundSel"][i].copy()

		#Get the radial posistions and velocities relative to the halo's bound particles
		r=np.sqrt(np.sum((partpos-meanpos)**2,axis=1))
		vr=np.sqrt(np.sum((partvel-meanvel)**2,axis=1))

		#Make any zeros be very small number so don't get errors when dividing by it
		r[r==0] = 1e-5 

		#Find the R_200crit and Mass_200crit from rhocrit
		R_200crit=cf.cosFindR200(r[TrackData["boundSel"][i]][r[TrackData["boundSel"][i]]<15*np.median(r[TrackData["boundSel"][i]])],rhocrit/unitinfo["Mass_unit"],200,GadHeaderInfo["partMass"]*unitinfo["Mass_unit"]/1e10)
		Mass_200crit=cf.coshaloRvirToMvir(R_200crit,rhocrit,200,Munit=unitinfo["Mass_unit"])

		#Calculate the escape velocity from the potential
		Vesc_calc=cf.cosNFWvesc(r, Mass_200crit, c=TrackData["Conc"][i], z = GadHeaderInfo["Redshift"],f=1,Munit=unitinfo["Mass_unit"],Dist="Ang")

		#Find the particles which are bound to the halo
		boundSel = (vr<Vesc_calc) & (r<R_200crit)
		npart=np.sum(boundSel)


		# Check if the halo has gone below the particle limit or is diffuse so its mass has gone above 2 times the initial mass from VELOCIraptor
		if((npart<=20) | (2*TrackData["Mvir"][i]<Mass_200crit)):


			if(TrackData["TrackedNsnaps"][i]>0):
				#Get the index of the WW halo in the previous snapshot
				progen=TrackData["progenitor"][i]
				progenIndx = int(TrackData["progenitor"][i]%opt.Temporal_haloidval-1)

				progenIndx = progenIndx - prevNhalo

				#Find the halo it has merged with if it has been lost using the particles that were previously bound to the halo
				if(TrackData["TrackDisp"][i]):
					MergeHalo(opt,treeOpt,meanpos,partIDs[TrackData["boundSel"][i]],progenIndx,snapdata,TrackData["host"][i],filenumhalos,pfiles,upfiles,grpfiles,prevappendTreeData,pos_tree,WWstat)

				#If not to be tracked until dispersed then connect it up with its endDesc
				else:
					prevappendTreeData["Descendants"][progenIndx] = TrackData["endDesc"][i]
					prevappendTreeData["NumDesc"][progenIndx] = 1

					#Since it is not possible to easily extract the endDesc particles (which could be many snapshots away)
					#the merit is just set to 1.0 for these halos
					prevappendTreeData["Merits"][progenIndx] = 1.0
					WWstat["ConnectPartLimit"]+=1
			else:
				WWstat["PartLimitStart"]+=1

			#Mark the halo to be deleted
			TrackData["idel"][i] = 1
			continue

		# Update the boundSel in the track data
		TrackData["boundSel"][i]=boundSel

		#Iterate on posistion until less than 100 particles or tracking less than 10% of the halos paricles
		j=0
		mbpsel = (vr/Vesc_calc<0.8/(2**j) ) & ((r/R_200crit)<0.8/(2**j))
		while(True):
			if((np.sum(mbpsel)<100) | ((np.sum(mbpsel)/npart)<0.1)):

				#Check if we have iterated to a core with less than 10 bound particles then return to the previous selection
				if((j>1) & (np.sum(mbpsel)<10)):
					# print("Iterated to core with less than 10 mpb",np.sum(mbpsel),"returning to the previous selection of",np.sum(tmpMBPsel),"from a",npart,"size halo")
					mbpsel = tmpMBPsel
				break

			tmpMBPsel = mbpsel
			mbpsel=(vr/Vesc_calc<0.8/(2**j) ) & ((r/R_200crit)<0.8/(2**j))
			j+=1

		# Update the mbpSel for the track data
		TrackData["mbpSel"][i] = mbpsel

		nmbpsel = np.sum(mbpsel)
		boundIDs = np.zeros(npart,dtype=np.uint64)

		#Order the IDs first the mbp radially sorted and then the less bound particles also radially sorted
		boundIDs[:nmbpsel] = partIDs[TrackData["mbpSel"][i]][np.argsort(r[TrackData["mbpSel"][i]])]
		boundIDs[nmbpsel:] = partIDs[(~TrackData["mbpSel"][i]) & (TrackData["boundSel"][i])][np.argsort(r[(~TrackData["mbpSel"][i]) & (TrackData["boundSel"][i])])]
		unboundIDs = partIDs[(~TrackData["mbpSel"][i]) & (~TrackData["boundSel"][i])]


		#Only try to match to a VELOCIraptor halo if this halo is to be tracked until it is dispersed
		if(TrackData["TrackDisp"][i]):

			#Find all the halos within 0.6 R200crit
			indx_list=np.sort(pos_tree.query_ball_point(meanpos,R_200crit))

			#Lets check if it has any matches
			if(indx_list.size):

				#Store all the WhereWolf merit, Matched merits and MatchedID's for 
				meritList = []
				MatchedIDList = []
				start=time.time()

				matched=False

				#Open up the first VELOCIraptor file
				fileno=0
				offset = 0

				#If doing core matching finc out how many particles to match in the core of the WW halo
				if((treeOpt["Core_fraction"]<1.0) & (treeOpt["Core_fraction"]>0.0)):
					WWCoreNpart = np.max([np.rint(treeOpt["Core_fraction"]*npart),treeOpt["Core_min_number_of_particles"]])
					WWCoreNpart = np.min([npart,WWCoreNpart]).astype(int)

				#Loop over all the matches
				for indx in indx_list:

					# if(halodata["hostHaloID"][indx]>-1):
					# 	continue

					MatchedID=snapdata["ID"][indx]
					if("Descen" in treedata.keys()):
						MatchedDesc = treedata["Descen"][indx]
						#Lets also extract the merit for the matched halo and lets see if this has a better match
					else:
						break

					#Add the halo to the check merged dict if it does not exist to see how long it has been phased mixed
					if(MatchedID not in TrackData["CheckMerged"][i].keys()):
						TrackData["CheckMerged"][i][MatchedID]=0

					#Loop over the VELOCIraptor files to find which one it is in
					while((indx+1)>(offset + filenumhalos[fileno])):
						offset+=filenumhalos[fileno]
						fileno+=1

					#Get the matched halo particles and properties
					matched_partIDs = WWio.GetHaloParticles(grpfiles[fileno],pfiles[fileno],upfiles[fileno],int(indx-offset))

					######## First match all particles to all particles  ########
					merit = CalculateMerit(treeOpt,boundIDs,matched_partIDs)

					#Only add it if the halo has no progenitor
					if(progenBool[indx]==False):
						meritList.append(merit)
						MatchedIDList.append(MatchedID)

					#Find the ratio of how far away the halo is to see if it is within 0.1Rvir
					ratioradius2 = ((meanpos[0]-snapdata["Xc"][indx])**2 + (meanpos[1]-snapdata["Yc"][indx])**2 + (meanpos[2]-snapdata["Zc"][indx])**2)/(snapdata["R_200crit"][indx]*snapdata["R_200crit"][indx])

					if((ratioradius2<0.1) & (snapdata["Mass_200crit"][indx]>0.5*Mass_200crit)):
						#If it is within 0.1 Rvir add one to the counter
						TrackData["CheckMerged"][i][MatchedID]+=1

						#If it is within 0.1 Rvir more than 3 times stop the tracking
						if(TrackData["CheckMerged"][i][MatchedID]==opt.NumSnapsWithinCoreMerge):

							progen=TrackData["progenitor"][i]

							progenIndx = int(TrackData["progenitor"][i]%opt.Temporal_haloidval-1)
							progenIndx = progenIndx - prevNhalo
							prevappendTreeData["Descendants"][progenIndx]= MatchedID
							prevappendTreeData["NumDesc"][progenIndx] = 1
							prevappendTreeData["Ranks"][progenIndx] = 1


							#Loop over the VELOCIraptor files to find which one it is in
							while((indx+1)>(offset + filenumhalos[fileno])):
								offset+=filenumhalos[fileno]
								fileno+=1
							#Calculate the merit between the halos
							prevappendTreeData["Merits"][progenIndx] = merit

							WWstat["Mixed"]+=1

							matched = True

							break

						#Now found a match need to update the ID in the CheckMerged dictonary to point to the descendant but only if not at the final snapshot
						if("Descen" in treedata.keys()):
							TrackData["CheckMerged"][i][MatchedDesc] = TrackData["CheckMerged"][i].pop(MatchedID)

					else:
						#If it goes outside of 0.1Rvir then delete the entry
						TrackData["CheckMerged"][i].pop(MatchedID)

				#The connection to the VELOCIraptor halo is made if the halo has no connection and the WhereWolf halo's merit is above the merit limit

				#First lets find which ones have no connection and are above the merit limit
				meritList=np.asarray(meritList,dtype=np.float32)
				Sel = meritList>treeOpt["Merit_limit_for_next_step"]

				#Find if there are any halos which satisfy the above conditions
				if(np.sum(Sel)):

					#Use a masked array to find the argmax
					maxindx = np.argmax(meritList)

					progen=TrackData["progenitor"][i]

					#Only update if this halo has been tracked for at least 1 snapshot
					if(TrackData["TrackedNsnaps"][i]>0):
						progenIndx = int(TrackData["progenitor"][i]%opt.Temporal_haloidval-1)
						progenIndx = progenIndx - prevNhalo
						prevappendTreeData["Descendants"][progenIndx]= MatchedIDList[maxindx]
						prevappendTreeData["NumDesc"][progenIndx] = 1
						prevappendTreeData["Merits"][progenIndx] = meritList[maxindx]
						WWstat["Matched"]+=1

					matched = True

				# #Remove the indexes where the MatchedMeritList==0 so where halos don't have any connections
				# meritList = meritList[MatchedMeritList>0]
				# MatchedIDList = MatchedIDList[MatchedMeritList>0]
				# MatchedMeritList = MatchedMeritList[MatchedMeritList>0]

				# #Since the halo has a connection(s) lets compare them and see if WhereWolf's connection is 50% better and if the existing merit is poor <0.1
				# Sel = (((meritList-MatchedMeritList)/MatchedMeritList)>0.5) & (MatchedMeritList<0.1)

				# #Find if there are any halos which satisfy the above conditions
				# if(np.sum(Sel)):

				# 	#Use a masked array to find the argmax
				# 	maxindx = meritList.argmax()

				# 	progen=TrackData["progenitor"][i]

				# 	print(meritList[maxindx],MatchedMeritList[maxindx],npart,MatchedIDList[maxindx])

				# 	#Only update if this halo has been tracked for at least 1 snapshot
				# 	if(TrackData["TrackedNsnaps"][i]>0):
				# 		progenIndx = int(TrackData["progenitor"][i]%opt.Temporal_haloidval-1)
				# 		progenIndx = progenIndx - prevNhalo
				# 		prevappendTreeData["Descendants"][progenIndx]= MatchedIDList[maxindx]
				# 		prevappendTreeData["NumDesc"][progenIndx] = 1
				# 		prevappendTreeData["Merits"][progenIndx] = meritList[maxindx]
				# 		WWstat["Switched"]+=1

				# 	matched = True

				if(matched):
					TrackData["idel"][i] = 1
					continue

		#From the caustic line can see if particles are bound to the halo
		ID=(snap+opt.Snapshot_offset)*opt.Temporal_haloidval +(len(snapdata["ID"]) + len(appendHaloData["ID"])) +1

		# Calculate properties of the halo
		R_200mean=cf.cosFindR200(r[TrackData["boundSel"][i]][r[TrackData["boundSel"][i]]<15*np.median(r[TrackData["boundSel"][i]])],rhomean/unitinfo["Mass_unit"],200,GadHeaderInfo["partMass"]*unitinfo["Mass_unit"]/1e10)
		Mass_200mean=cf.coshaloRvirToMvir(R_200mean,rhomean,200,Munit=unitinfo["Mass_unit"])
		Mass_tot = npart * GadHeaderInfo["partMass"]*unitinfo["Mass_unit"]/1e10
		sigV = np.mean(np.sqrt((vr-np.mean(vr))**2))
		Vmax = vr[vr.argmax()]
		Rmax = r[r.argmax()]

		# Now lets append the data to the lists
		appendHaloData["ID"].append(ID)
		appendHaloData["Mass_200crit"].append(Mass_200crit)
		appendHaloData["Mass_200mean"].append(Mass_200mean)
		appendHaloData["Mass_tot"].append(Mass_tot)
		appendHaloData["R_200crit"].append(R_200crit)
		appendHaloData["R_200mean"].append(R_200mean)
		appendHaloData["npart"].append(npart)			
		appendHaloData["Xc"].append(meanpos[0])
		appendHaloData["Yc"].append(meanpos[1])
		appendHaloData["Zc"].append(meanpos[2])
		appendHaloData["VXc"].append(meanvel[0])
		appendHaloData["VYc"].append(meanvel[1])
		appendHaloData["VZc"].append(meanvel[2])
		appendHaloData["sigV"].append(sigV)
		appendHaloData["Vmax"].append(Vmax)
		appendHaloData["Rmax"].append(Rmax)
		appendHaloData["Group_Size"].append(npart)
		appendHaloData["Particle_IDs"].extend(boundIDs.tolist())
		appendHaloData["Offset"].append(appendHaloData["Offset"][-1]+npart)
		appendHaloData["Offset_unbound"].append(0)
		appendHaloData["Num_of_groups"]+=1

		#Lets store the previous host of this halo
		prevhost = TrackData["host"][i]

		#Lets see if the halo has a host if not then we can try and set one
		if((TrackData["host"][i]>0) & (int(TrackData["host"][i]/opt.Temporal_haloidval)==snap+opt.Snapshot_offset)):
				appendHaloData["Parent_halo_ID"].append(int(TrackData["host"][i]%opt.Temporal_haloidval+1))
				appendHaloData["hostHaloID"].append(TrackData["host"][i])

				#Only need to do this if not at the final snapshot
				if("Descen" in treedata.keys()):
					HostIndx=int(TrackData["host"][i]%opt.Temporal_haloidval -1)
					hostHeadID=treedata["Descen"][HostIndx]

					#Lets check this host halo is in thr next snapshot
					hostHeadSnap = int(hostHeadID/opt.Temporal_haloidval)

					#Update the host in the trackdata
					if((TrackData["host"][i]!=hostHeadID) & (hostHeadSnap==snap+1)): 
						TrackData["host"][i]=hostHeadID
					else:
						TrackData["host"][i]=-1
		else:
			#Check if the halo has a new host by first searching the 100 closest halos
			dists,indexes=pos_tree.query(meanpos,100)

			# Loop over the closet matches
			for dist,indx in zip(dists,indexes):
				hostR200 = snapdata["R_200crit"][indx]
				hostM200 = snapdata["Mass_200crit"][indx]

				#Check if the WhereWolf halo is within the viral radius of the halo
				ratioradius = dist/hostR200
				if(ratioradius<1.0):

					#If it is within the viral radius check if it is gravitationally bound
					v2=np.sum(meanvel**2)
					boundCal=0.5 * Mass_200crit * v2 - (G * hostM200 * Mass_200crit)/dist 
					if((boundCal<0.0) & (Mass_200crit<hostM200)):
						#If gravitationall bound then set is as the host and subtract the mass from it of the WW halo
						appendHaloData["hostHaloID"].append(snapdata["ID"][indx])
						appendHaloData["Parent_halo_ID"].append(indx+1)

						#Lets uppdate the host to point to this halos head if its in the next snapshot
						if("Descen" in treedata.keys()):
							HostIndx=int(snapdata["ID"][indx]%opt.Temporal_haloidval -1)
							hostHeadID=treedata["Descen"][HostIndx]

							#Lets check this host halo is in thr next snapshot
							hostHeadSnap = int(hostHeadID/opt.Temporal_haloidval)
							if(hostHeadSnap==snap+1):
								TrackData["host"][i]=hostHeadID
							else:
								TrackData["host"][i] = -1

						break

			#If there was nothing found set it to -1
			else:
				appendHaloData["hostHaloID"].append(-1)
				appendHaloData["Parent_halo_ID"].append(-1)

		#Set the descendant of this halos progenitor to point to this halo
		progen=TrackData["progenitor"][i]
	

		#Add in a entry in the tree for this halo
		appendTreeData["ID"].append(ID)
		appendTreeData["Descendants"].append(np.int64(0))# Will be updated in the next snapshot
		appendTreeData["Ranks"].append(0)
		appendTreeData["NumDesc"].append(0)
		appendTreeData["Merits"].append(0)

		# #For debugging
		# appendTreeData["endDesc"].append(TrackData["endDesc"][i])

		if(TrackData["TrackedNsnaps"][i]==0):
			prevupdateTreeData["ID"].append(progen)
			prevupdateTreeData["Descendants"].append(ID)

			#Calculate its merit with the particles in the original VELOCIraptor halo 
			#to the ones which are currently bound to the halo
			merit = CalculateMerit(treeOpt,partIDs[TrackData["boundSel"][i]],partIDs)
			prevupdateTreeData["Merits"].append(merit)

		else:

			progenIndx = int(progen%opt.Temporal_haloidval-1)

			progenIndx = progenIndx - prevNhalo 
			prevappendTreeData["Descendants"][progenIndx]= ID
			prevappendTreeData["NumDesc"][progenIndx] = 1

			#Now calculate the merit between the particles that were previously bound to the halo
			#to the ones that were previously bound
			merit = CalculateMerit(treeOpt,partIDs[TrackData["boundSel"][i]],partIDs[prevboundSel])
			prevappendTreeData["Merits"][progenIndx] = merit

		#Set this halo as the halo in the next snapshot progenitor
		TrackData["progenitor"][i] = ID

		#Iterate how many snaps it has been tracked for
		TrackData["TrackedNsnaps"][i]+=1


		#Lets see if the halo has reached the last snapshot it should be tracked for
		if((int(TrackData["endDesc"][i]/opt.Temporal_haloidval)==snap+opt.Snapshot_offset+1) & (TrackData["TrackDisp"][i]==False) & ("Descen" in treedata.keys())):

			appendTreeData["Descendants"][-1] = TrackData["endDesc"][i]
			appendTreeData["NumDesc"][-1] = 1

			#Since it is not possible to easily extract the endDesc particles (which could be many snapshots away)
			#the merit is just set to 1.0 for these halos
			appendTreeData["Merits"][-1] = 1.0
			WWstat["Connect"]+=1

			TrackData["idel"][i] = 1

			continue


		# Check if the halo still has a bound center of at least 5 particles
		if((np.sum(TrackData["mbpSel"][i])<=5) & ("Descen" in treedata.keys())):
		
				
			progenIndx = -1

			#Find the halo it has merged with if it has been lost
			if(TrackData["TrackDisp"][i]):
				 MergeHalo(opt,treeOpt,meanpos,partIDs[TrackData["boundSel"][i]],-1,snapdata,prevhost,filenumhalos,pfiles,upfiles,grpfiles,appendTreeData,pos_tree,WWstat)
				 WWstat["MergedMBP"]+=1

			#If not to be tracked until dispersed then connect it up with its endDesc
			else:
				appendTreeData["Descendants"][-1] = TrackData["endDesc"][i]
				appendTreeData["NumDesc"][-1] = 1

				#Since it is not possible to easily extract the endDesc particles (which could be many snapshots away)
				#the merit is just set to 1.0 for these halos
				appendTreeData["Merits"][-1] = 1.0
				WWstat["ConnectMBP"]+=1

			#Mark the halo to be deleted
			TrackData["idel"][i] = 1
			continue

		WWstat["contNSnap"]+=1

		#Update the keepPIDS array so the PID is kept
		keepPIDS[partOffset:partOffset+totnpart]=True

		#Lets also alter the order of the particles in the allpid array
		npart = np.uint64(npart)
		allpid[partOffset:partOffset+npart] = boundIDs
		allpid[partOffset+npart:partOffset+totnpart] = unboundIDs

		#Update the offsets
		newPartOffsets.append(np.uint64(newPartOffset))

		#Add to the offset for the next halo
		newPartOffset+=totnpart

	nextpids = allpid[keepPIDS].copy() #Need to be copy as will be used by startTrack()

	#Update the TrackData by removing all the halos to be removed
	for i in range(len(TrackData["progenitor"])-1,-1,-1):
		if(TrackData["idel"][i]):
			for field in TrackData.keys():
				del TrackData[field][i]




	del pos_tree

	return newPartOffsets,nextpids



