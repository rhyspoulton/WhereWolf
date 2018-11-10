import numpy as np
import WWio

#Function to populate giant array with all the merits 
def CheckHaloHasProgen(opt,treeOpt,isnap,istep,tree,numhalos,allnumhalos,ihalostart,ihaloend,progenBool):

	updatedindexes = [[] for i in range(istep)]
	merits  = [[] for i  in range(istep)]

	for i in range(isnap,isnap+istep+1):
		if(len(progenBool[i])==0):
			progenBool[i] = np.zeros(allnumhalos[i],dtype=bool)
	#For each primary descendant lets populate the position of its descendant with the merit to its progenitor
	for i in range(numhalos):

		#See if this halo is the is the primaray descendant
		if(tree["Rank"][ihalostart:ihaloend][i]==0):

			#If so lets extract its descedant snap and index to populate the progenBool
			snap = int(tree["Descen"][ihalostart:ihaloend][i]/opt.Temporal_haloidval) - opt.Snapshot_offset
			index = int(tree["Descen"][ihalostart:ihaloend][i]%opt.Temporal_haloidval-1)
			if(snap>opt.numsnaps-1):
				continue
			progenBool[snap][index]=True

			#Added to the update list so can be communicated to other processes
			updatedindexes[snap-isnap-1].append(index)
	for i in range(istep):
		updatedindexes[i] = np.asarray(updatedindexes[i],dtype=int)

	return updatedindexes

#Calculate the merit between two haloes that have been matched
def CalculateMerit(treeOpt, partList1, partList2, icore=True):

	sel=np.in1d(partList1,partList2)
	nsh=np.sum(sel,dtype=np.float64)
	n1 = len(partList1)
	n2 = len(partList2)

	#Lets check if this connection is insignificant to the poisson noise
	#If so return a merit of zero so it will not be matched up
	if((nsh<treeOpt["Merit_limit"]*np.sqrt(n1)) & (nsh<treeOpt["Merit_limit"]*np.sqrt(n2))):
		return 0.0;

	if(treeOpt["Merit_type"]==1):
		merit=nsh*nsh/n1/n2
	elif (treeOpt["Merit_type"]==2):
		merit=nsh/n1
	elif (treeOpt["Merit_type"]==3):
		merit=nsh
	elif (treeOpt["Merit_type"]==4):
		merit=nsh/n1*(1.0+nsh/n2)
	elif (treeOpt["Merit_type"]==5):
		#this ranking is based on Poole+ 2017 where we rank most bound particles more
		#assumes input particle list is meaningful (either boundness ranked or radially for example
		ranksum= np.sum(1.0/(np.where(sel)[0]+1.0))
		#normalize to the optimal value for nsh=n2, all particles in the descendant
		norm=0.5772156649+np.log(n2)
		merit=ranksum/norm
		#and multiply this to standard Nsh^2/N1/N2 merit to correct for small objects
		#being lost in bigger object and being deemed main progenitor
		merit*=nsh*nsh/n1/n2
		merit=np.sqrt(merit)
	elif (treeOpt["Merit_type"]==6):
		#like above but ranking both ways, that is merit is combination of rankings in a and b
		ranksum=0
		ranksum= np.sum(1.0/(np.where(sel)[0]+1.0))
		#normalize to the optimal value for nsh=n2, all particles in the descendant
		norm=0.5772156649+np.log(n1)
		merit=ranksum/norm

		#Now do the ranking the other way
		sel2 = np.in1d(partList2,partList1)
		ranksum=np.sum(1.0/(np.where(sel2)[0]+1.0))
		norm=0.5772156649+np.log(n2)
		merit*=ranksum/norm
		merit*=nsh*nsh/n1/n2
		merit=np.sqrt(merit)

	######### Then do core to core if the core fractoion is between 0 to 1  ########
	if((treeOpt["Core_fraction"]<1.0) & (treeOpt["Core_fraction"]>0.0) & icore):

		#Find the number of core particles to use
		CoreNpart1 = np.max([np.rint(treeOpt["Core_fraction"]*n1),treeOpt["Core_min_number_of_particles"]])
		CoreNpart1 = np.min([n1,CoreNpart1]).astype(int)

		#First do a baised core to all
		coremerit = CalculateMerit(treeOpt,partList1[:CoreNpart1],partList2,False)
		if(coremerit>merit):
			merit=coremerit

		CoreNpart2 = np.max([np.rint(treeOpt["Core_fraction"]*n2),treeOpt["Core_min_number_of_particles"]])
		CoreNpart2 = np.min([n2,CoreNpart2]).astype(int)

		#Calculate the core merit using just a unbaised Nsh^2/N1/N2 merit
		sel=np.in1d(partList1[:CoreNpart1],partList2[:CoreNpart2])
		nsh = np.sum(sel,dtype=np.float64)
		coremerit = nsh*nsh/CoreNpart1/CoreNpart2

		#Now lets check if this merit is any better, if so then update the merit
		if(coremerit>merit):
			merit=coremerit


	return merit

def MergeHalo(opt,treeOpt,meanpos,partIDs,progenIndx,snapdata,host,filenumhalos,pfiles,upfiles,grpfiles,prevappendTreeData,pos_tree,WWstat):

	#Lets find the num halos search of halos closest to this halo
	if(opt.Num_Halos_search>snapdata["ID"].size):
		_,indx_list = pos_tree.query(meanpos,snapdata["ID"].size)
	else:
		_,indx_list = pos_tree.query(meanpos,opt.Num_Halos_search)

	meritList = np.zeros(opt.Num_Halos_search,dtype=float)
	match=False
	#Lets find if the halo lies within any of the rvir of the halos
	for i,indx in enumerate(indx_list):

		fileno = 0
		offset = 0
		while((indx+1)>(offset + filenumhalos[fileno])):
			offset+=filenumhalos[fileno]
			fileno+=1

		#Get the matched halo particles and properties
		matched_partIDs = WWio.GetHaloParticles(grpfiles[fileno],pfiles[fileno],upfiles[fileno],int(indx-offset))

		#Do a merit calculation to see if they do overlap
		meritList[i]=CalculateMerit(treeOpt,partIDs,matched_partIDs)

	# HostIndx = int(host%opt.Temporal_haloidval-1)
	# fileno = 0
	# offset = 0
	# while((HostIndx+1)>(offset + filenumhalos[fileno])):
	# 	offset+=filenumhalos[fileno]
	# 	fileno+=1

	# host_partIDs = WWio.GetHaloParticles(grpfiles[fileno],pfiles[fileno],upfiles[fileno],int(HostIndx-offset))
	# matchNpart = host_partIDs.size
	# npart = partIDs.size

	# #Find the amount of particles that match in the two halos
	# sel=np.in1d(host_partIDs,partIDs)
	# nsh=np.sum(sel,dtype=np.float64)

	# hostMerit=(nsh*nsh)/(matchNpart*npart)

	#Only need to update the descendant if a match is found
	if np.sum(meritList)!=0:


		#The one which has the highest merit is the best match
		maxIndx = np.argmax(meritList)
		indx = indx_list[maxIndx]

		#Lets have the previous halo point to the matched halo
		prevappendTreeData["Descendants"][progenIndx] = snapdata["ID"][indx]
		prevappendTreeData["Ranks"][progenIndx] = 1
		prevappendTreeData["NumDesc"][progenIndx] = 1 
		prevappendTreeData["Merits"][progenIndx] = meritList[maxIndx]

		WWstat["Merged"]+=1

	elif(host!=-1):

		HostIndx = int(host%opt.Temporal_haloidval-1)
		fileno = 0
		offset = 0
		while((HostIndx+1)>(offset + filenumhalos[fileno])):
			offset+=filenumhalos[fileno]
			fileno+=1

		host_partIDs = WWio.GetHaloParticles(grpfiles[fileno],pfiles[fileno],upfiles[fileno],int(HostIndx-offset))

		merit = CalculateMerit(treeOpt,partIDs,host_partIDs)

		#If it does have a host lets just merge it with it
		prevappendTreeData["Descendants"][progenIndx] = host
		prevappendTreeData["Ranks"][progenIndx] = 1
		prevappendTreeData["NumDesc"][progenIndx] = 1 
		prevappendTreeData["Merits"][progenIndx] = merit

		WWstat["Merged"]+=1

		#Else if all merits are 0 then there are no matches found so keep NumDesc==0

	else:
		WWstat["notMerged"]+=1

