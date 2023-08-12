from networks import FeedForwardNN
import numpy as np
import torch
from torch import multiprocessing as mp
import logging 



def convert_to_probs(numbers_list):
    # Convert the list of numbers to a NumPy array
    numbers_array = np.array(numbers_list)
    
    # Apply the softmax function to convert the numbers to probabilities
    exp_numbers = np.exp(numbers_array)
    probabilities = exp_numbers / np.sum(exp_numbers)
    
    return probabilities

class PPO:
	def __init__(self, env, learning_rate, load_previus_models=False) -> None:
		logging.debug(f'Device: {"cuda" if torch.cuda.is_available() else "cpu"}')
		logging.debug(f"Load previus models: {load_previus_models}")
		logging.debug(f"Learning rate: {learning_rate}")
		self.env = env
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		

		self.actor = FeedForwardNN(self.env.num_inputs, self.env.num_actions).to(self.device)
		self.critic = FeedForwardNN(self.env.num_inputs, 1).to(self.device)
		
		if load_previus_models:
			print("...loading...")
			self.actor.load_state_dict(torch.load("models/actor.pt"))
			self.critic.load_state_dict(torch.load("models/critic.pt"))


		self.cov_var = torch.full(size=(self.env.num_actions, ), fill_value=0.5).to(self.device)
		# Create the covariance matrix
		self.cov_mat = torch.diag(self.cov_var).to(self.device)
		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
	
	def get_action(self, obs):
		
		mean = self.actor(torch.tensor(obs, dtype=torch.float32).to(self.device))	# Same thing as calling self.actor.forward(obs)
		dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)		# Create our Multivariate Normal Distribution
		action = dist.sample()								# Sample an action from the distribution and get its log prob
		log_prob = dist.log_prob(action)
		
		return action.detach().cpu().numpy(), log_prob.detach()

	def compute_rewards_to_go(self, batch_rewards, gamma=0.9995):

		batch_rtgs = []	
		# Iterate through each episode backwards to maintain same order in batch_rtgs
		for ep_rews in reversed(batch_rewards):		
			discounted_reward = 0 # The discounted reward so far		
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * gamma
				batch_rtgs.insert(0, discounted_reward)	# Convert the rewards-to-go into a tensor

		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)	
		return batch_rtgs
	
	def evaluate(self, batch_obs, batch_acts):
		# Query critic network for a value V for each obs in batch_obs.
		V = self.critic(batch_obs).squeeze()	

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)	# Return predicted values V and log probs log_probs
		return V, log_probs

	
	def colect_data(self, timesteps_per_batch, max_timesteps_per_episode, database):
		while True:
			# Batch data
			batch_obs = []			 # batch states
			batch_acts = []			# batch actions
			batch_log_probs = []		 # log probs of each action
			batch_rews = []			# batch rewards
			batch_rtgs = []			# batch rewards-to-go
			batch_lens = []			# episodic lengths in batch
			
			obs = self.env.reset()
			done = False
			# Number of timesteps run so far this batch
			t = 0 
			while t < timesteps_per_batch:	# Rewards this episode
				ep_rews = []	
				obs = self.env.reset()
				done = False	
				

				ep_t = 0
				
				for ep_t in range(max_timesteps_per_episode):
					
					t += 1				# Increment timesteps ran this batch so far
					batch_obs.append(obs)		# Collect observation
					
					action, log_prob = self.get_action(obs)
					
					action_one = np.random.choice(action.size, p = convert_to_probs(action))
					obs, rew, done, _ = self.env.step(action_one)
					
					# Collect reward, action, and log prob
					ep_rews.append(rew)
					batch_acts.append(action)
					batch_log_probs.append(log_prob)	
					if done:
						break	# Collect episodic length and rewards

				batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
				batch_rews.append(ep_rews)
				
			
			batch_rtgs = self.compute_rewards_to_go(batch_rews)

			#adds data to queue
			database.put([torch.tensor(batch_obs, dtype=torch.float32).to(self.device), torch.tensor(batch_acts, dtype=torch.float32).to(self.device), torch.tensor(batch_log_probs, dtype=torch.float32).to(self.device), torch.tensor(batch_rtgs, dtype=torch.float32).to(self.device), torch.tensor(batch_lens, dtype=torch.float32).to(self.device)])

		


	def learn(self, num_epochs, max_timesteps_per_episode, updates_per_epoch, epsilon):
		logging.debug(f"Number of epochs: {num_epochs}")
		logging.debug(f"Max timesteps per episode: {num_epochs}")
		logging.debug(f"Updates per epoch: {updates_per_epoch}")
		logging.debug(f"Epsilon: {epsilon}")
		t = 0
		losses_actor = []
		losses_critic = []
		processes = []

		database = mp.Queue()
		logging.debug(f"Number of cores: {mp.cpu_count()}")
		for _ in range(mp.cpu_count() - 1):
			process = mp.Process(target=self.colect_data, args=(num_epochs, max_timesteps_per_episode, database))
			process.start()
			

		while t < num_epochs:
			
			while database.qsize() < 1:
				pass
			
			if t % 100 == 99:
				logging.debug("Saving models")
				torch.save(self.actor.state_dict(), "models/actor.pt")
				torch.save(self.critic.state_dict(), "models/critic.pt")
			
			data_list = database.get()
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = data_list
			del data_list

			# Calculate V_{phi, k}
			V, _ = self.evaluate(batch_obs, batch_acts)
			# Calculate advantage
			A_k = batch_rtgs - V.detach()
			# normalize
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
			for _ in range(updates_per_epoch):
				# Calculate pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate ratios
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * A_k
				actor_loss = (-torch.min(surr1, surr2)).mean()
				losses_actor.append(actor_loss.item())

				logging.info(f"Actor loss: {actor_loss}")

				critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

				logging.info(f"Critic loss: {critic_loss}")
				
				
				
				losses_critic.append(critic_loss.item())

				# Calculate gradients and perform backward propagation for actor 
				# network
				self.actor_optim.zero_grad()
				actor_loss.backward()
				self.actor_optim.step()
				
				# Calculate gradients and perform backward propagation for critic network    

				self.critic_optim.zero_grad()    
				critic_loss.backward()    
				self.critic_optim.step()

			del batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
				

			t += 1
		return losses_actor, losses_critic