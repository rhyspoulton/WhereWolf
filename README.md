# WhereWolf

Developer: Rhys Poulton

This is a code to continue to track halos (or galaxies) after they have been lost by the halo finder. For the results of using this see Poulton et al., submitted. For further results and a detailed description of how the code works see Poulton et al. in prep.

## Running

This code is run by typing:

mpirun -np \<# of cpus> python wherewolf.py -c wherewolf.cfg -s \<numsnaps> -o \<output directory>

Note that WhereWolf is currently heavily I/O bound so there will be little difference in the timings for lots of MPI threads. Please see the example configuation file and update the full paths to the input filelist.

## Output

WhereWolf output files differenly for the VELOCIraptor halo catalogue and the TreeFrog merger-tree.

### VELOCIraptor

WhereWolf will add halos to the VELOCIrator catalogue by each mpi thread adding on a additional mpi file and will update the information in the original VELOCIraptor files

### TreeFrog

For TreeFrog, WhereWolf will create a new tree file for each snapshot that halos have been ghosted for. These files will have a .WW.tree extension. A file containing the names of the tree per snapshot will be created in \<output directory>/treesnaplist.txt.

### WhereWolf run statistics

As a diagnostic check WhereWolf also creates a file containing the statistics of why halos terminate tracking per snapshot. This can be useful in comparing different halo catalogues. This file is created in \<output directory>/WWrunstat.txt.

## Notes

This code is still currently under development so please contact me if you run into any problems or you think something can be improved
