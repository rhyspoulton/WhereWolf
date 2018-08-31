import numpy as np 
from mpi4py import MPI

def UpdateIDsoffsets(comm,Rank,size,snap,appendHaloData,prevupdateTreeData,appendTreeData,prevappendTreeData,prevNhalo,HALOIDVAL=1000000000000):




	#Have the root process gather all the offsets
	if(Rank==0):
		IDOffsets = np.zeros(size,dtype=np.uint64)
		IDOffsets[1] = len(appendHaloData["ID"])

		for iprocess in range(1,size-1):
			ntracked = comm.recv(source=iprocess,tag=1)

			#Compute the
			IDOffsets[iprocess+1] = ntracked + IDOffsets[iprocess]

	else:
		IDOffsets = None
		comm.send(len(appendHaloData["ID"]),dest=0,tag=1)



	#Now we got all the offsets lets send it back to the processes
	if(Rank==0):

		for iprocess in range(1,size):

			comm.send(IDOffsets[iprocess],dest=iprocess,tag=2)

		IDOffset = IDOffsets[0]
	else:

		IDOffset = comm.recv(source=0,tag=2)

	#Update the ID's in appendHaloData
	appendHaloData["ID"] += IDOffset

	#Update the data in prevupdateTreeData where the link is to a WW halo
	sel = (prevupdateTreeData["Descendants"]%HALOIDVAL-1).astype(int,copy=False) >= prevNhalo
	prevupdateTreeData["Descendants"][sel] += IDOffset

	#Update the ID's in appendTreeData
	appendTreeData["ID"] += IDOffset

	if(prevappendTreeData is not None):
		sel = ((prevappendTreeData["Descendants"]%HALOIDVAL-1).astype(int,copy=False) >= prevNhalo) & ((prevappendTreeData["Descendants"]/HALOIDVAL).astype(int,copy=False)==snap)
		prevappendTreeData["Descendants"][sel] += IDOffset


	return appendHaloData,prevupdateTreeData,appendTreeData,prevappendTreeData




def GatheroutputTreeData(comm,Rank,size,appendTreeData,prevupdateTreeData,treeDtype):

	if(appendTreeData is not None):
		AllAppendTreeData = {}
		for field in appendTreeData.keys():

			appendTreeData[field] = np.asarray(appendTreeData[field])

			#Let the non-root process send the ammunt of data that the root-process is expected to recieve
			if(Rank==0):
				istart = 0
				iend = len(appendTreeData[field])
				ALLnappend = [len(appendTreeData[field])]

				# Lets first tell the root-process how much data to expect from each process
				for iprocess in range(1,size):
					nappend = comm.recv(source=iprocess,tag=11)
					ALLnappend.append(nappend)

				# Find the total amount to send
				NumSend = np.sum(ALLnappend)

				# Allocate an array to store all the tree data
				AllAppendTreeData[field] = np.empty(NumSend,dtype=treeDtype[field])

				# Add the data from the root process
				AllAppendTreeData[field][istart:iend] = appendTreeData[field]

				for iprocess in range(1,size):
					istart=iend
					iend+=ALLnappend[iprocess]

					#Recieve buffere
					tmp = np.empty(ALLnappend[iprocess],dtype=treeDtype[field])

					# Recieve the data from the other processes
					comm.Recv(tmp,source=iprocess,tag=21)

					# Add it into the array
					AllAppendTreeData[field][istart:iend] = tmp





			else:
				ALLnappend = None
				AllAppendTreeData = None

				# Send the data to the root process
				nappend = len(appendTreeData[field])
				comm.send(nappend,dest=0,tag=11)
				comm.Send(appendTreeData[field],dest=0,tag=21)

	else:
		AllAppendTreeData = appendTreeData

	if(prevupdateTreeData is not None):

		AllUpdateTreeData = {}

		for field in prevupdateTreeData.keys():

			prevupdateTreeData[field] = np.asarray(prevupdateTreeData[field])

			#Let the non-root process send the ammunt of data that the root-process is expected to recieve
			if(Rank==0):
				istart = 0
				iend = len(prevupdateTreeData[field])
				ALLnupdate = [len(prevupdateTreeData[field])]

				# Lets first tell the root-process how much data to expect from each process
				for iprocess in range(1,size):
					nappend = comm.recv(source=iprocess,tag=11)
					ALLnupdate.append(nappend)

				# Find the total amount to send
				NumSend = np.sum(ALLnupdate)

				# Allocate an array to store all the tree data
				AllUpdateTreeData[field] = np.empty(NumSend,dtype=np.uint64)

				# Add the data from the root process
				AllUpdateTreeData[field][istart:iend] = prevupdateTreeData[field]

				for iprocess in range(1,size):
					istart=iend
					iend+=ALLnupdate[iprocess]

					#Recieve buffere
					tmp = np.empty(ALLnupdate[iprocess],dtype=np.uint64)

					# Recieve the data from the other processes
					comm.Recv(tmp,source=iprocess,tag=21)
					# print("Recieving from",iprocess,tmp)

					# Add it into the array
					AllUpdateTreeData[field][istart:iend] = tmp


			else:
				ALLnupdate = None
				AllUpdateTreeData = None

				# Send the data to the root process
				nappend = len(prevupdateTreeData[field])
				comm.send(nappend,dest=0,tag=11)
				# print(Rank,"Sending",prevupdateTreeData[field])
				comm.Send(prevupdateTreeData[field],dest=0,tag=21)
	else:
		AllUpdateTreeData=prevupdateTreeData


	return AllAppendTreeData,AllUpdateTreeData



