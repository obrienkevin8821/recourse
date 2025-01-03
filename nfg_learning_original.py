import random
# import matplotlib.pyplot as plt
import numpy as np

num_runs = 1
num_episodes = 20000
num_agents = 2
num_actions = 2
agent_payoffs = []
action_history = []
agent_prob_action1 = []
alpha = 0.1
epsilon = 0.3
# softmax_temp =
decay_alpha = True
decay_epsilon = True
alpha_decay_rate = 0.999
epsilon_decay_rate = 0.999


# simple NFG for testing purposes
# payoffs = [
#      [[0, 0], [0, 0]],
#      [[0, 0], [1, 1]]
# ]

# another simple NFG for testing purposes
# payoffs = [
#     [[4, 4], [3, 3]],
#     [[2, 2], [1, 1]]
# ]

# matching pennies NFG
payoffs = [
     [[1, -1], [-1, 1]],
     [[-1, 1], [1, -1]]
]

# stag hunt NFG
# payoffs = [
#      [[5, 5], [0, 2]],
#      [[2, 0], [1, 1]]
# ]

# rock paper scissors NFG
# payoffs = [
#     [[0, 0], [-1, 1], [1, -1]],
#     [[1, -1], [0, 0], [1, -1]],
#     [[-1, 1], [1, -1], [0, 0]]
# ]


def get_payoffs(selected_actions):
    return payoffs[selected_actions[0]][selected_actions[1]]


def do_episode(agent_list):
    selected_actions = []
    for agent in agent_list:
        selected_actions.append(agent.select_action())
        # print("selected_actions", selected_actions)
    received_payoffs = get_payoffs(selected_actions)
    print("selected_actions", selected_actions)
    print("received_payoffs", received_payoffs)

    for a in range(len(agent_list)):
        print("before update: agent", a, "action values",  agent_list[a].action_values)
        agent_list[a].update(selected_actions[a], received_payoffs[a])
        print("after update: agent", a, "action values",  agent_list[a].action_values)

    return selected_actions, received_payoffs


class Agent:
    def __init__(self, a, e):
        self.alpha = a
        self.epsilon = e
        self.action_values = [0 for a in range(num_actions)] # produces [0, 0], initial actions to take for both agents
        #print("self.action_values", self.action_values)

    def select_action(self):
        #return self.epsilon_greedy()
        return self.select_softmax()

    def random_action(self):
        return random.randrange(num_actions) # returns a number less than 2, so 0 or 1

    def greedy_action(self):
        max_value = -100000000
        best_action = -1
        for a in range(num_actions):
            if self.action_values[a] > max_value:
                max_value = self.action_values[a]
                best_action = a
        return best_action

    def epsilon_greedy(self):
        if random.random() < self.epsilon: # random.random() returns a number between 0.1 and 1.0
            return self.random_action()
        else:
            return self.greedy_action()

    def get_softmax_strategy(self):
        #TODO use softmax to calcualte the probability distribution over actions
        return []

    def select_softmax(self):
        #TODO use softmax
        tau = 10.2
        q_values = np.array(self.action_values)
        exp_x = np.exp(q_values/tau)
        exp_x /= np.sum(exp_x)
        return np.random.choice(len(q_values), p=exp_x)
        # ...


    def update(self, action, payoff):
        #print(action, payoff)
        self.action_values[action] = self.action_values[action] + alpha * (payoff - self.action_values[action])


def main():
    for run in range(num_runs):
        print("----- Beginning run", run, "-----")
        agent_list = []
        agent_payoffs.append([])
        action_history.append([])
        for agent in range(num_agents):
            agent_list.append(Agent(alpha, epsilon))

        for episode in range(num_episodes):
            print("     *** Episode", episode, "***")
            do_episode(agent_list)
            if decay_alpha:
                for agent in agent_list:
                    agent.alpha = agent.alpha * alpha_decay_rate
            if decay_epsilon:
                for agent in agent_list:
                    agent.epsilon = agent.epsilon * epsilon_decay_rate

        print("run over")
        print("agent", 0, "action values", agent_list[0].action_values)
        print("agent", 1, "action values", agent_list[1].action_values)
        final_tau = 0.1
        final_actions = np.exp(np.array(agent_list[0].action_values)/final_tau)
        final_actions /= np.sum(final_actions)
        print("Agent 0 final actions:", final_actions)
        final_actions = np.exp(np.array(agent_list[1].action_values)/final_tau)
        final_actions /= np.sum(final_actions)
        print("Agent 1 final actions:", final_actions)        


if __name__ == "__main__":
    main()