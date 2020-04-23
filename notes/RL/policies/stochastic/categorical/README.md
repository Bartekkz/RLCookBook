## What is Categorical Policy?
Categorical policy is like a classifier over discrete actions (e.g. action_space = (1, 2, 3, 4)).
You build a neural netowork model for categorical policy the same as You would build it for typical supervised learning  classification task. For example:  
  
• input is the observation  
• then we have a neural network model with some architecture (Linear layers, Conv layers, etc.)
   depending on the input  
• last layer in the network is the Linear layer that gives you *logit for each action  
• finally we have a *softmax function to convert logits into probabilities  

## Where Categorical Policies are used?
Categorical policies are used in discrete action spaces.

## Sampling actions from policy
As said in the main [note](https://github.com/Bartekkz/RLCookBook/blob/master/notes/policies/stochastic/README.md) for stochastic policies sampling actions in one of the key computations while using and training stochastic policies. In categorical policy we sample from policy using *__Categorical Distribution__.
## References
#### logits  
This term might be confusing, because in math it means something a little bit diffrent than in ML.  
In __math__, Logit is a function that maps probabilities ([0, 1]) to R( (-inf, inf) ) with the following formulas:  
![](https://i.stack.imgur.com/zto5q.png)  
Important to note is that: Probability of 0.5 corresponds to a logit of 0. Negative logit correspond to probabilities less than 0.5, positive to > 0.5.

When in __ML__ it can be the vector or raw (non-normalized) predictions that a cliassifier generated, which are then passsed to a normalization function. In mutli-class classification problems, logits typically become and input to the softmax function.

If You want to get a deeper understanding about diffrent definitons of logits, You can read this [stackoverflow post](https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow) which should clarify Your doubts.
#### Softmax
Softmax function is often used in neural networks. It maps the non-normalized output of a network to a probability distribution over predicted output classes. The standard softmax funxtion is defined by following formula:
![](https://i.stack.imgur.com/iP8Du.png), where 'i' is the number of classes.
#### Categorical Distribution
The term 'categorical distribution' has come to mean two seperate things. __Informally__, it is any
distribution with categories. __Alternatively__(and more precisely), it is a __generalization__ of the [Bernoulli distribution](https://www.statisticshowto.com/categorical-distribution) for categorical random variable. While a random variable in Bernoulli distribution has two possible outcomes, a categorical random variable
has multiple possibilites. Additional when there is a single trial, the categorical distribution is equal to a multinomial
distribution. #[Reference](https://www.statisticshowto.com/categorical-distribution)
