#Base OpenAI code from: https://github.com/openai/gym/blob/master/examples/scripts/sim_env
#Base OpenCV code from: https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
#Other stuff: https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym
#!/usr/bin/env python
import gym                      #Environment
import numpy as np              #Used for linear algebra
import tensorflow as tf
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
f = open("DeepLearn.txt", "a")

#Seed in order to reproduce results
#random.seed(0)
#np.random.seed(0)

#Factors for Q
numEps = 4000
discountFactor = 0.99
learningRate = 0.15
reportInterval = 500
exploration = lambda ep: 50. / (ep + 10)    #Probability the agent moves randomly

#Converts variables into a form for machine learning
def encode(i, n):
    return np.identity(n)[i].reshape((1, -1))

def printReport(title, rewards, episode):
    f.write("%s\tAverage per 100 eps: %.2f\tBest Average of 100 eps: %.2f\tOverall ep average: %.2f\tEpisode: %d\n" % (title, np.mean(rewards[-100:]), max([np.mean(rewards[i:i+100]) for i in range(len(rewards)-100)]), np.mean(rewards), episode))

def main():
    #Create the environment and get the available actions
    env = gym.make(args.env)
    env.seed(0)

    #Keep a running total of reward value
    rewards = []

    #Placeholders for algorithm data
    nObserves, nActions = env.observation_space.n, env.action_space.n
    observeTPH = tf.placeholder(shape=[1,nObserves], dtype=tf.float32)
    observeTNPH = tf.placeholder(shape=[1,nObserves], dtype=tf.float32)
    actPH = tf.placeholder(tf.int32, shape=())
    rewPH = tf.placeholder(shape(), dtype=tf.float32)
    qTargetPH = tf.placeholder(shape[1, nActions], dtype=tf.float32)

    #Begin calculations by calculating current and target Q
    W = tf.Variable(tf.random_uniform([nObserves, nActions], 0, 0.01))
    qCurrent = tf.matmul(observeTPH, W)
    qTarget = tf.matmul(observeTNPH, W)

    qTargetMax = tf.reduce_max(qTargetPH, axis=1)
    qTargetSA = rewPH + discountFactor * qTargetMax
    qCurrentSA = qCurrent[0, actPH]
    error = tf.reduce_sum(tf.square(qTargetSA - qCurrentSA))
    predictActPH = tf.argmax(qCurrent, 1)                       #Action that maximizes Q

    #Optimizer
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    updateModel = trainer.minimize(error)

    #Initialize Tensorflow session and variables for computations
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for ep in range(1, numEps + 1):
            observeT = env.reset()
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
                    #Calculate action with some probability of randomness
                    observeTExplore = explore(observeT, nObserves)
                    action = session.run(predictActPH, feed_dict={observeTPH: observeTExplore})
                    if np.random.rand(1) < exploration(ep):
                        action = env.action_space.sample()

                    observeTN, reward, done, _ = env.step(action)

                    #Train Neural Network
                    observeTNExplore = explore(observeTN, nObserves)
                    qTargetValue = session.run(qTarget, feed_dict={observeTNPH: oberseTTNExplore})
                    session.run(updateModel, feed_dict={observeTPH: observeTExplore, rewPH: reward, qTargetPH: qTargetValue, actPH: action})

                    epReward += reward
                    observeT = observeTN
                    print("\n%d" % ep)
                    env.render()
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
