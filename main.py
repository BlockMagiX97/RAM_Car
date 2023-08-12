import ppo
import CarGameEnviroment
import gym
import time
import matplotlib.pyplot as plt
import matplotlib
import torch
import logging

if __name__ == '__main__':
    logging.basicConfig(filename='run.log', 
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s') 
    logging.debug("Started") # for identifing separate starts of training program
    time_s = time.time()
    env = CarGameEnviroment.CarGameEnviroment()
    torch.multiprocessing.set_start_method('spawn')
    agent = ppo.PPO(env, 0.000001)
    losses_actor, losses_critic = agent.learn(10000, 20, 5, 0.2)

    # plotting graphs

    plt.figure()
    plt.plot(range(len(losses_actor)), losses_actor, marker='o', linestyle='-', color='b', label='Data')


    # Add labels and title
    plt.xlabel('timesteps')
    plt.ylabel('loss')
    plt.title('Actor')

    # Save the plot to a file
    plt.savefig('actor.png')

    # Close the plot (optional, but recommended)
    plt.close()

    plt.figure()
    plt.plot(range(len(losses_critic)), losses_critic, marker='o', linestyle='-', color='b', label='Data')


    # Add labels and title
    plt.xlabel('timesteps')
    plt.ylabel('loss')
    plt.title('Critic')

    # Save the plot to a file
    plt.savefig('critic.png')

    # Close the plot (optional, but recommended)
    plt.close()

    print(time.time() - time_s)