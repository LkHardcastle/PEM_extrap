# Sampler
# - Setup
# Inner loop:
# - Find next deterministic time
# - Evaluate potential and time-gradient of potential
# - If potential is decreasing in time, line search for minimum and update to get minimum break if next deterministic time reached
# - Simulate v = rand() update sampler to next point exactly break if next deterministic time reached
# - Backfill sampled points

# Sampler function runs set up starts inner loop and runs storage after each event etc.
# sampler_inner:
### - Find next deterministic time and evaluate at that time
### - Initial evaluation
######### - Line search for 0 if necessary
######### - Line search for next event 
### - Return skeleton point
### - Update split, merge, stick, unstick times
## Storage etc.
