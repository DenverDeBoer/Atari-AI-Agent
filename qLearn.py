#Authors:
#   Denver DeBoer
#   Alex Woods
#   Kevin Holkeboer
#Description:
#   This uses OpenAI Gym to create a Q learning algorithm for FrozenLake
#   It uses a Q table to store various states and cooresponding actions
#Base OpenAI code from: https://github.com/openai/gym/blob/master/examples/scripts/sim_env
#Q-learning: https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym

#!/usr/bin/env python
import gym                      #Environment
import numpy as np              #Used for linear algebra
import random                   #Allow for seeding

import argparse                 #Get command line arguments
from builtins import input      #Used for getting user input

from time import sleep          #To pause the agent

#Get arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("env")
parser.add_argument("-m", "--mode", choices=["random", "human", "agent"],
    default="agent")
args = parser.parse_args()

#Open a file to write report to
f = open("Learn.txt", "a")

#Seed in order to reproduce results
#random.seed(0)
#np.random.seed(0)

#Factors for Q
numEps = 5000
discountFactor = 0.85
learningRate = 0.9
reportInterval = 1000

def printReport(title, rewards, episode):
    f.write("%s\tAverage per 100 eps: %.2f\tBest Average of 100 eps: %.2f\tOverall ep average: %.2f\tEpisode: %d\n" % (title, np.mean(rewards[-100:]), max([np.mean(rewards[i:i+100]) for i in range(len(rewards)-100)]), np.mean(rewards), episode))

def main():
    #Create the environment and get the available actions
    env = gym.make(args.env)
    env.seed(0)

    #Keep a running total of reward value
    rewards = []

    #Q-value table that will hold relationships between actions and reward
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for ep in range(1, numEps + 1):
        state = env.reset()
        epReward = 0
        env.render(mode='human')
        while True:
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
                        sleep(2)
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
                noise = np.random.random((1, env.action_space.n)) / (ep**2.)
                action = np.argmax(Q[state, :] + noise)
                nextState, reward, done, _ = env.step(action)
                QTarget = reward + discountFactor * np.max(Q[nextState, :])
                Q[state, action] = (1-learningRate) * Q[state, action] + learningRate * QTarget
                epReward += reward
                state = nextState

                #Print the game
#                print("\n%d" % ep)
#                env.render()

                if done:
                    rewards.append(epReward)
                    if ep % reportInterval == 0:
                        printReport("Agent", rewards, ep)
                        sleep(2)
                    break
    printReport("Overall", rewards, -1)
    f.write("\n\n")
    f.close()

if __name__ == '__main__':
    main()
