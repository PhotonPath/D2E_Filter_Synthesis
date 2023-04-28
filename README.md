# D2E_Filter_Synthesis
Here a brief descriptions of the files in this repository.
1. Lattice n Filter
   This script is used as a benchmark. It creates a Lattice of any order with the Photonic Building Block library and you can use it to play with the 
   Lattice heater to better understand them. 
   
2. Partial Madsen
   This script replicates the Madsen Algorithm given both output bar and cross profiles as An, Bn polynomials
   
3. Complete Madsen
   This script replicates the Madsen Algorithm given the cross profile as An polynomial
   
4. Madsen with Ideal Lattice
   This script uses the results of the Madsen Algorithm on a Lattice generated by the Photonic Building Block library. The lattice is ideal in the sense that
   you can change the coupler values as wanted by the Madsen. The profile is given as An polynomial.
   
5. and 6. Madsen with Balance
   These scripts try to apply the results of the Madsen Algorithm on a Lattice generated by the Photonic Building Block library. The lattice is less ideal in      the sense that the coupler are formed by a (COUPLER - BALANCE - COUPLER) block. The profile is given as An polynomial. FAIL.
   
7. Madsen BackForth
   This script tries to apply the results of the Madsen Algorithm on a Lattice generated by the Photonic Building Block library. The lattice is less ideal in      the sense that the coupler are formed by a (COUPLER - BALANCE - COUPLER) block. The profile is given as An polynomial. SUCCEESS.
   
8. Madsen BackForth from Target
   This script tries to apply the results of the Madsen Algorithm on a Lattice generated by the Photonic Building Block library. The lattice is less ideal in      the sense that the coupler are formed by a (COUPLER - BALANCE - COUPLER) block. The profile is given a frequency function.







asd









dsad






9. Madsen BackForth from Target
   This script tries to apply the results of the Madsen Algorithm on a Lattice generated by the Photonic Building Block library. The lattice is less ideal in      the sense that the coupler are formed by a (COUPLER - BALANCE - COUPLER) block. The profile is given a frequency function. Different non-idealities are          considered, such as propagation losses, coupling losses, coupler not 50%.
   
   -5- Interactive Madsen BackForth from Target
   This scripts allow to use the final Madsen BackForth Algorithm in an interactive way. Please not that here no propagation and coupling losses are                considered. The difference between the two interactive scripts is from where the target is given (zeros of the target polynomial or magnitude shape of the
   target) and the possibility of changing the slope of the coupler.
