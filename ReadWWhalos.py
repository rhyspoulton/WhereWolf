import numpy as np 
import h5py
import os


"""

Function to read and append the wherewolf halos to the VELOCIraptor catalogue

"""

def ReadandAppendWhereWolfHalos(basefilename,numhalos,halodata,iverbose=False,desiredfields=[]):
	"""
	WhereWolf has a standard naming convention so the basefilename needs to be: /path/to/WhereWolf/output/snapshot_###
	"""

	#First check if the file exists
	filext = ".VELOCIraptor.WW.properties."
	filename = basefilename + filext + "0"
	if(os.path.isfile(filename)==False):
		print("WhereWolf file not found at this snapshot")
		return


	if(iverbose): print("Reading",filename)
	#Open up the file and see if is a WhereWolf catalog
	hdffile = h5py.File(filename,"r")
	if("WWfile" not in hdffile.attrs.keys()):
		print("The catalog does not have the 'WWfile' flag indicting it is a WhereWolf catalog")
		hdffile.close()
		return

	#Can extract the information to read the rest of the WhereWolf catalog
	numfiles = int(hdffile["Num_of_files"][...])
	WWnumhalos = np.uint64(hdffile["Total_num_of_groups"][0])

	#Find the fall the fields in the hdffile
	fieldnames = [str(n) for n in hdffile.keys()]
	# clean of header info
	fieldnames.remove("File_id")
	fieldnames.remove("Num_of_files")
	fieldnames.remove("Num_of_groups")
	fieldnames.remove("Total_num_of_groups")
	fieldtype = [hdffile[fieldname].dtype for fieldname in fieldnames]

	#Check which fields to load 
	if (len(desiredfields) > 0):
		if (iverbose):
			print("Loading subset of all fields in WhereWolf property file ",len(desiredfields), " instead of ", len(fieldnames))
		fieldnames = desiredfields
		fieldtype = [hdffile[fieldname].dtype for fieldname in fieldnames]

	hdffile.close()

	#See if all of the fields exist in the halodata
	for field in fieldnames:
		if(field not in halodata.keys()):
			print("The field",field,"does not exist in the VELOCIraptor's halodata please check the correct fields have been read in")

	#Lets generate a updated halodata that also included the WhereWolf halos
	totnumhalos = numhalos + WWnumhalos
	allhalodata = {fieldnames[i]: np.zeros(totnumhalos, dtype=fieldtype[i]) for i in range(len(fieldnames))}

	#Insert the current halodata into allhalodata
	for field in fieldnames:
		allhalodata[field][:numhalos] = halodata[field]

	#Loop over all the files in the WhereWolf catalog and add to the allhalodata
	noffset = numhalos
	for i in range(numfiles):
		filename = basefilename + filext + str(i)
		hdffile = h5py.File(filename,"r")
		numhalos = np.uint64(hdffile["Num_of_groups"][0])
		for field in fieldnames:
			allhalodata[field][noffset:noffset+numhalos]=np.asarray(hdffile[field])
		noffset+=numhalos

	#Extract the simulation and unit info if it exists
	if("SimulationInfo" in halodata.keys()):
		allhalodata["SimulationInfo"]=halodata["SimulationInfo"]

	if("UnitInfo" in halodata.keys()):
		allhalodata["UnitInfo"]=halodata["UnitInfo"]

	return allhalodata




