import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.nn import softmax

from ramo.strategy.strategies import softmax_strategy


class IndependentActorCriticAgent:
    """An independent learner using the multi-objective actor-critic algorithm for the SER criterion.

    This implementation is based on the multi-objective actor-critic algorithm proposed in [1].

    References:
        .. [1] Zhang, Y., Rădulescu, R., Mannion, P., Roijers, D., & Nowé, A. (2020). Opponent Modelling for
            Reinforcement Learning in Multi-Objective Normal Form Games. In Proceedings of the 19th International
            Conference on Autonomous Agents and MultiAgent Systems (pp. 2080–2082). International Foundation for
            Autonomous Agents and Multiagent Systems.

    """
    def u_linear(w):
        return lambda p: (p[0] * w) + (p[1] * (1-w))

    w1_row = 1.0
    w1_col = 0.0
    u_row = u_linear(w1_row)
    u_col = u_linear(w1_col)
    u_tpl = (u_row, u_col) # depending on agent self.u = (u_row) or (u_col), or maybe just leave as self.u = u_tpl
   
    self.u = u_tpl
    self.grad = jit(grad(self.objective_function))
        

    self.alpha_theta = alpha_theta # 0.01
    self.alpha_theta_decay = alpha_theta_decay # 1

    self.theta = np.zeros(num_actions)
    self.policy = softmax_strategy(self.theta)




    def objective_function(self, theta, q_values):
        """The objective function for the agent. This is the SER criterion.

        Args:
            theta (ndarray): The policy parameters.
            q_values (ndarray): The expected returns for the actions.

        Returns:
            float: The utility from the current policy and Q-values.

        """
        policy = softmax(theta)
        expected_returns = jnp.matmul(policy, q_values)
        utility = self.u(expected_returns)
        return utility

    def update(self, action, reward):
        """Perform an update for the agent.

        Args:
            action (int): The actions that was taken by the agent.
            reward (float): The reward that was obtained by the agent.

        Returns:

        """
        self.update_q_table(action, reward)
        self.theta += self.alpha_theta * self.grad(self.theta, self.q_table)
        self.policy = softmax_strategy(self.theta)
        self.update_parameters()

    def update_q_table(self, action, reward):
        """Update the Q-table based on the chosen actions and the obtained reward.

        Args:
            action (int): The action chosen by this agent.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        self.q_table[action] += self.alpha_q * (reward - self.q_table[action])

    def update_parameters(self):
        """Update the hyperparameters. Decays the learning rate for the Q-values and policy parameters."""
        self.alpha_q *= self.alpha_q_decay
        self.alpha_theta *= self.alpha_theta_decay

    def select_action(self):
        """Select an action according to the agent's policy.

        Returns:
            int: The selected action.

        """
        return self.rng.choice(range(self.num_actions), p=self.policy)
