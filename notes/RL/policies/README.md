## What are policy gradients methods?
As You probably already know the goal of reinforcement learning is to find an optimal behavior strategy for the agent to obtain optimal rewards. The __policy gradient__ methods target at modeling and optimizing the policy directly. The policy is usually modeled with parametrized function respect to θ, πθ(a|s). The value of the reward (objective) function depends on this policy and we can apply various algorithms to optimize θ. Firstly let's have a look at how reward function is defined:  
![](https://github.com/Bartekkz/RLCookBook/blob/master/resources/PG_Reward_Function.png)  
, where __dπ__ denotes the discounted probability of reaching each
state under the policy __π__.  
As said before we want to optimize θ in order to change our policy
to maximize the return. We can achieve with a help of a __gradient ascent__ by moving θ toward the direction suggested by the gradient ∇θJ(θ).
