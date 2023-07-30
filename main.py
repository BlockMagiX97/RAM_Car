from CarGameEnviroment import CarGameEnviroment
import numpy as np
from agent import Agent


if __name__ == '__main__':
    env = CarGameEnviroment()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.num_actions, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  )
    agent.load_models()
    n_games = 30000

    figure_file = 'plots/cartpole.png'

    best_score = float("-inf")
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        if i % 100 == 9:
            agent.save_models()
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action,
                                   prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]

