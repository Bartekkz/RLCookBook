## Stochastic policies
### Most used stochastic policies
The two most common kinds of stochastic policies in deep RL are __categorical policies__ and
__diagonal Gaussian policies__.
### Key computations for using and training stochastic policies
* __sampling__ actions from the policy
* computing __log likelihoods__ of particaular actions.  
In subdirections I discuss both __sampling__ and __log likelihoods__ for each stochastic policy.

### Why we use __importance sampling__ (approximation)?
Sampling from the policy to learn introduce an invalid bias to the
estimation of the gradient. That's why we use importance sampling to correct the distribution. Moreover once You use importance sampling
with the gradient, You can simplify the result into the gradient of
the log probability

### What are __log likelihoods__?
First, let consider what is the objective function we want to maximize? One can be:
 