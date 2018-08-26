# Udacity Reinforcement Learning: Navigation Project
### Introduction
This project contains an implementation of DQN (Deep Q-Network) model for reinforcement learning problem of banana collector in unity as part of udacity deep reinforcement course . In the game  agents move around an area attempting to collect as many rewarding bananas (yellow) as possible, while avoiding negatively rewarding bananas (purple). The objective of the DQN model is to use be able to train an agent to maximize reward (yellow banana while avoiding purple banana) leveraging information on the state (continous) and making discrete decisiosn (direction of movement).

### Input to the Model
The agent can observe a state space with 37 dimensions that includes the agent's velocity and ray-based perception of objects around the agents forward direction. These are obtained from the environment provided by unity. We are leveraging a single agent in this environment.
The actions that the agent can take are discrete including moving forward, backward, and turning left or right.

### Key Files In the Project and parameters to be adjusted
There are 3 files in the project file:

1. Navigation.ipynb: Jupyter notebook file which you can use to load the environment and train the model. As we are using linux environment it is headless, to visualize the output you can download the weights which are output of the model and use it windows or other supported OS to see how the trained model works.
To setup the environment from unity for the banana collector please download the relevant zip file (e.g. as in linux environment used this code please use the link: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip). Additionally we will need to install packages needed as indicated in the jupyter notebook. 

The code loads in environment and trains the DQN model until the average reward over the recent 100 episodes is >=13.0 which was provided as the benchmark to equal or exceed in this challenge.

2. dqn_agent.py: Python file that contains the functions to train the agent using Q-learning. The beginning of the file also contains some of the key hyperparameters for Q-learning including replay buffer size, batch size for training, update frequency of the target network etc.

3. model.py: Python file that contains function which creates the Neural Network for the DQN. The architecture of the neural network model is defined in this file (# of layers and number of units in each layer). We use pytorch to build the neural network based model.

### Output of the project
We have two outputs from the jupyter notebook:
1. Trained NN model weights: The model weights after the training is done is stored as output: checkpoint.pth
2. Graph of the average reward over 100 episodes: The graph shows how the model improves over the training and the graphs ends with the episode when average value over last 100 episods >=13.
