# Q-learning
Quick facts about Q-learning:  
    - online  
    - uses boostrap learning  
    - off policy  
## Tabular Q-learning
Just for now we are gonna stick to the most basic form of Q-learning, which as title says is tabular form.
For the problems with small number of possible states we can create a Q-table. 
### What is Q-table?
Q-table is a matrix (n-dimensional array) which stores values of all avaiable <state, action> pairs in given environment.
### How we update Q-table?
We update Q-table with following formula: 
![equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686)
As we see in formula we use old estimates and new estimates to update q-value. This kind of updating is called 
Bootstraping
### Exploration vs Exploitation
In our example we used epsilon greedy strategy to deal with exploration vs exploitation problem, which means that with probability
1 - epsilon we will take greedy action and with probability epsilon we will take random action in order to explore 

