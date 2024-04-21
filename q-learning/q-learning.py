'''
Q-learning Algorithm
Implemented for Gymnasium Taxi Environment
'''
import numpy as np
import gymnasium as gym
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import pygame


class QLearningParams:
    def __init__(self, learning_rate=0.5, gamma=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon


class QLearningAgent:
    def __init__(self, env: gym.Env, params: QLearningParams, strategy):
        self.env = env
        self.params = params
        self.state_space_size = env.observation_space.n
        self.action_space_size = env.action_space.n
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.strategy = strategy

    def epsilon_greedy_action(self, state, is_training=True):
        if np.random.rand() < self.params.epsilon and is_training:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state, :])

    def boltzmann_action(self, state, temperature):
        action_values = self.q_table[state, :]
        normalized_values = action_values - np.max(action_values)
        exp_values = np.exp(normalized_values / temperature)
        action_probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(
            np.arange(self.action_space_size), p=action_probabilities
        )

    def choose_action(self, state, episode=None, is_training=True):
        if self.strategy == "epsilon_greedy":
            return self.epsilon_greedy_action(state, is_training)
        elif self.strategy == "boltzmann":
            temperature = 1.0 / (episode + 1)
            return self.boltzmann_action(state, temperature)
        else:
            raise ValueError("Invalid strategy")

    def update_q_table(self, state, action, next_state, reward):
        self.q_table[state, action] = self.q_table[
            state, action
        ] + self.params.learning_rate * (
            reward
            + self.params.gamma * np.max(self.q_table[next_state, :])
            - self.q_table[state, action]
        )

    def eval_self_winratio(self, tries: int) -> int:
        wins = 0
        for curr_try in range(tries):
            observation, _ = self.env.reset()
            done = False
            truncated = False
            while not done and not truncated:
                action = self.choose_action(observation, curr_try, False)
                next_observation, _, done, truncated, _ = self.env.step(action)
                observation = next_observation
                if done and not truncated:
                    wins += 1
        return wins

    def visualize_route(self) -> None:
        self.strategy = "epsilon_greedy"
        environment = gym.make("Taxi-v3", render_mode="human")
        curr_state, _ = environment.reset()
        terminal = False
        truncated = False
        while not terminal and not truncated:
            environment.render()
            action = self.choose_action(curr_state, is_training=False)
            next_state, _, terminal, truncated, _ = environment.step(action)
            curr_state = next_state
            pygame.time.delay(100)
        environment.render()
        environment.close()


class QLearningTrainer:
    def __init__(self, env: gym.Env, params: QLearningParams):
        self.env = env
        self.params = params

    def train_agents(
        self, num_episodes
    ) -> tuple[dict[str, QLearningAgent], list[float]]:
        strategies_to_test = ["epsilon_greedy", "boltzmann"]
        total_rewards = {strategy: [] for strategy in strategies_to_test}
        agents = dict()

        for strategy in strategies_to_test:
            agent = QLearningAgent(self.env, self.params, strategy)
            for episode in range(num_episodes):
                observation, _ = self.env.reset()
                done = False
                truncated = False

                while not done and not truncated:
                    action = agent.choose_action(observation, episode)
                    next_observation, reward, done, truncated, _ = self.env.step(action)
                    agent.update_q_table(
                        int(observation), action, int(next_observation), reward
                    )
                    observation = next_observation

                total_rewards[strategy].append(agent.q_table.sum())
            agents[strategy] = agent

        return agents, total_rewards


def plot_hiperparameter_evaluation(
    trainer: QLearningTrainer, param_name, total_rewards
):
    strategies_to_test = ["epsilon_greedy", "boltzmann"]

    for strategy in strategies_to_test:
        plt.plot(
            range(len(total_rewards[strategy])),
            total_rewards[strategy],
            label=f"{strategy} Strategy",
            marker=".",
        )
        if param_name == "epsilon":
            break

    plt.title(
        f"Hiperparameter {param_name} influence on Total reward across Learning Episodes,Params={trainer.params.__dict__}",
        wrap=True,
    )
    plt.xlabel("Learning Episodes")
    plt.ylabel("Total reward")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    env = gym.make("Taxi-v3", max_episode_steps=50)
    hiperparams = ["epsilon"]
    strategies = ["epsilon_greedy", "boltzmann"]
    params_range = np.arange(0.1, 1.0, 0.1)

    num_episodes = 20000

    for param in hiperparams:
        for value in params_range:
            kwargs = {param: value}
            params = QLearningParams(**kwargs)
            trainer = QLearningTrainer(env, params)
            _, results = trainer.train_agents(num_episodes)
            plot_hiperparameter_evaluation(trainer, results)

    episodes_to_test = [1250, 2500, 5000]

    win_tries = 50
    ticks = np.arange(0, win_tries + 0.01, 10)

    for param in hiperparams:
        fig, axis = plt.subplots(1, 3)
        for episodes, ax in zip(episodes_to_test, axis):
            results = {strat: [] for strat in strategies}

            for value in params_range:
                kwargs = {param: value}
                params = QLearningParams(**kwargs)
                trainer = QLearningTrainer(env, params)
                agents, _ = trainer.train_agents(episodes)

                for strat, agent in agents.items():
                    wins = agent.eval_self_winratio(win_tries)
                    results[strat].append(wins)

            for i, strat in enumerate(strategies):
                width = 0.05
                ax.bar(
                    params_range + width / 2 * i,
                    results[strat],
                    width=width,
                    label=f"{strat} strategy",
                )
                if param == "epsilon":
                    break
            ax.legend()
            ax.set_title(f"Results for {episodes} episodes", wrap=True)
            ax.set_yticks(ticks=ticks)
            ax.set_xticks(ticks=params_range)
            ax.grid(True)
            ax.set_axisbelow(True)

        fig.text(0.5, 0.01, "Learning rate", ha="center", va="center")
        fig.text(0.08, 0.5, "Wins", ha="center", va="center", rotation="vertical")
        fig.suptitle(f"Impact of {param} parameter on Q-algorithm win-ratio")
        fig.set_figwidth(15)
        fig.savefig(f"wins_{param}.pdf")
        plt.close()

    env.close()


if __name__ == "__main__":
    main()
