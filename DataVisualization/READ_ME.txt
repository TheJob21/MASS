The onedrive folder below includes spectrum data and a basic script to load, process, and plot it. 
There are two data files that include 1M length-1024 snapshots worth of data each:
spectrum_245ghz.dat is raw complex64 IQ data from 2.4-2.5 GHz 
 spectrum_264ghz.dat is raw complex64 IQ data from 2.59-2.69 GHz 

The sample rate is 100 Msamps/sec, so each length-1024 snapshot occurs over 10.24 us of real time. 
process_spectrum.py shows how to go from time-domain I/Q samples to spectrum detections.

10,240 usec
1000 snapshots = 800 simulated timesteps