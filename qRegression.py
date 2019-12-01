#Authors:
#   Denver DeBoer
#   Alex Woods
#   Kevin Holkeboer
#Description:
#   This code utilizes OpenAI Gym to create a Q-learning agent
#   It uses linear regression to estimate various game states and appropriate cooresponding actions
#Base OpenAI code from: https://github.com/openai/gym/blob/master/examples/scripts/sim_env
#Q learning: https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym

#!/usr/bin/env python
import gym                      #Environment
import numpy as np              #Used for linear algebra
import random                   #Allow for seeding

import argparse                 #Get command line arguments
from builtins import input      #Used for getting user input

from time import sleep

#Get arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("env")
parser.add_argument("-m", "--mode", choices=["random", "human", "agent"],
    default="agent")
args = parser.parse_args()

#Open a file to write report to
f = open("Regression.txt", "a")

#Seed in order to reproduce results
#random.seed(0)
#np.random.seed(0)

#Factors for Q
numEps = 5000
discountFactor = 0.85
learningRate = 0.9
wLearningRate = 0.5
reportInterval = 1000

#Abstract the model
def makeQ(model):
    return lambda X: X.dot(model)

#Initialize the model
def initialize(shape):
    W = np.random.normal(0.0, 0.1, shape)
    Q = makeQ(W)
    return W, Q

#Train the model
def train(X, Y, W):
    i = np.eye(X.shape[1])
    newW = np.linalg.inv(X.T.dot(X) + 10e-4 * i).dot(X.T.dot(Y))
    W = wLearningRate * newW + (1 - wLearningRate) * W
    Q = makeQ(W)
    return W, Q

#Convert variables for machine learning
def encode(i, n):
    return np.identity(n)[i]

def printReport(title, rewards, episode):
    f.write("%s\tAverage per 100 eps: %.2f\tBest Average of 100 eps: %.2f\tOverall ep average: %.2f\tEpisode: %d\n" % (title, np.mean(rewards[-100:]), max([np.mean(rewards[i:i+100]) for i in range(len(rewards)-100)]), np.mean(rewards), episode))

def main():
    #Create the environment and get the available actions
    env = gym.make(args.env)
    env.seed(0)

    #Keep a running total of reward value
    rewards = []

    #Setup for regression model
    nObserves, nActions = env.observation_space.n, env.action_space.n
    W, Q = initialize((nObserves, nActions))
    states, labels = [], []

    for ep in range(1, numEps + 1):
        #If states or labels are full then reset them
        if len(states) >= 10000:
            states, labels = [], []

        state = encode(env.reset(), nObserves)
        epReward = 0
        env.render(mode='human')
        while True:
            states.append(state)
            #Performs random action from list of possible actions
            if args.mode == "random":
                action = env.action_space.sample()
                _, reward, done, _ = env.step(action)
                epReward += reward
                env.render()
                if done:
                    rewards.append(epReward)
                    if ep % reportInterval == 0:
                        printReport("Random", rewards, ep)
                        sleep(1)
                    break
            #Player controlled by user input
            elif args.mode == "human":
                action = input("type action from {0,...,%i} and press enter: "%(env.action_space.n-1))
                try:
                    action = int(action)
                    if action >= env.action_space.n:
                        print("Illegal action '{}'.".format(a))
                        action = 0
                    env.render()
                except ValueError:
                    print("WARNING: ignoring illegal action '{}'.".format(a))
                    action = 0
            #AI Agent ideally better than random agent and faster than human
            elif args.mode == 'agent':
                noise = np.random.random((1, nActions)) / ep
                action = np.argmax(Q(state) + noise)
                nextState, reward, done, _ = env.step(action)
                nextState = encode(nextState, nObserves)

                QTarget = reward + discountFactor * np.max(Q(nextState))
                label = Q(state)
                label[action] = (1 - learningRate) * label[action] + learningRate * QTarget
                labels.append(label)

                epReward += reward
                state = nextState

                #Train the model periodically
#                if len(states) % 10 == 0:
#                    W, Q = train(np.array(states), np.array(labels), W)

                #Print the game
                print("\n%d" % ep)
                env.render()

                if done:
                    rewards.append(epReward)
                    if ep % reportInterval == 0:
                        printReport("Agent", rewards, ep)
                        sleep(1)
                    break
    printReport("Overall", rewards, -1)
    f.write("\n\n")
    f.close()

if __name__ == '__main__':
    main()
