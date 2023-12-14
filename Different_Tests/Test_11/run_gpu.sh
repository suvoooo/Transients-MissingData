universe = vanilla 

executable = job_gpu.sh

log = logs/test.log.$(Cluster)-$(Process) 
error = logs/test.err.$(Cluster)-$(Process) 

request_Gpus = 1
request_Cpus = 4
request_Memory = 200GB
#+UseNvidiaA100 = True

#+testJob = True

queue

