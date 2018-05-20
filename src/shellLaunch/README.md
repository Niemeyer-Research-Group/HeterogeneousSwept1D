## Shell Scripts To Run Cluster Experiments and Tests

### Notes: 
* If your executables are compiled with the: NO OUTPUT flag there will be no equation solution output, only timing output.
* Timing output occurs for ALL runs and all scripts here use the trslt or rslt folders for output. A new run will append timing data to the end of the last 
run's csv file if that file has not been removed and processed. 
* All standard output is directed to the [RUNNAME].out file in the .runOut folder in the top directory.

### Scripts

__hSweep__ - Main Test.  Evaluates all equations using swept and classic on a dense grid of threads per block, gpu affinity, and grid size values.  This is the 
canonical test of the results.

__hAffinity__ - Secondary Test.  Evaluates all variations on dense grid of thread per block, sparse grid with larger range of gpu affinity, and a few sparse 
values of grid size.  This test is designed to show what ranges of gpu affinity and threads per block are most likely to produce the best performance.  The 
canonical test is then updated with these ranges to avoid wasting time testing the programs with execution parameters we know are bad.

__testall__ - Runs each variation once using middling execution parameters to make sure it works.

__testit__ - Runs ONCE and takes 2 or 3 arguments with the final argument in quotes: testit.sh [Equation (Euler)] [Scheme (Swept)] '[Additional Args] (tpb 128 
gpuA 10.0 nX 55555555)' 

