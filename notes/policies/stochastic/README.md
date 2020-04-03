# What is Categorical Policy?
Categorical policy is like a classifier over discrete actions (e.g. action_space = (1, 2, 3, 4)).
You build a neural netowrk model for categorical policy the same as You would build it for typical classifier for supervised learning task. For example:  
  
• input is the observation  
• then we have a neural network model with some architecture (Linear layers or Conv layers)
   depending on the input  
• last layers is the Linear layer that gives you *logits for each action  
• finally we have a *softmax to convert logits into probabilities  
