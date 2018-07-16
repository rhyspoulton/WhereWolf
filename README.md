# WhereWolf

Developer: Rhys Poulton

This is a code to continue to track halos (or galaxies) after they have been lost by the halo finder. For the results of using this see Poulton et al., submitted. For further results and a detailed description of how the code works see Poulton et al. in prep.

## Running

This code is run by typing:

mpirun -np <# of cpus> python wherewolf.py -c wherewolf.cfg -s <numsnaps> -o <output directory>

Note that WhereWolf is currently heavily I/O bound so there will be little difference in the timings for lots of MPI threads. Please see the example configuation file and update the full paths to the input filelist.  
 
## Notes

This code is still currently under development so please contact me if you run into any problems or you think something can be improved
