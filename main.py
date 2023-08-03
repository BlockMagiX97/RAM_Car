import ppo
import CarGameEnviroment
import gym
import time
import matplotlib.pyplot as plt
import matplotlib



time_s = time.time()
env = CarGameEnviroment.CarGameEnviroment()

agent = ppo.PPO(env, 0.000001)
losses_actor, losses_critic = agent.learn(10000, 20000, 5, 0.2)

plt.figure()
plt.plot(range(len(losses_actor)), losses_actor, marker='o', linestyle='-', color='b', label='Data')


# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sample Data Plot')

# Save the plot to a file
plt.savefig('actor.png')

# Close the plot (optional, but recommended)
plt.close()

plt.figure()
plt.plot(range(len(losses_critic)), losses_critic, marker='o', linestyle='-', color='b', label='Data')


# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sample Data Plot')

# Save the plot to a file
plt.savefig('critic.png')

# Close the plot (optional, but recommended)
plt.close()

print(time.time() - time_s)