# amended version attempting actor critic implementation.
# Just have it for 2 action NFG for now - see u_linear function below - line 24.
# The critic produces a value for theta which is used by the actor to update its policy.
# Theta is calculated based on a gradient. The gradient is the direction towards a desired result.
# Theta is recalculated on every episode.
# Code for actor critic is found with comment ## AC ##

import random
import matplotlib.pyplot as plt
import numpy as np 
import nashpy as nash # used for calculating nash


## AC Start ##
# required for actor critic - "pip install jax"
import jax.numpy as jnp
from jax import grad, jit
from jax.nn import softmax

# required for actor critic - "pip install ramo"
from ramo.strategy.strategies import softmax_strategy

# required for actor critic - utility function for single objective, 2 player, 2 action game
def u_linear(w):
    return lambda p: (p[0] * w) + (p[1] * (1-w))

w1_row = 1.0
w1_col = 0.0
u_row = u_linear(w1_row)
u_col = u_linear(w1_col)
u_tpl = (u_row, u_col)
## AC End ##

num_runs = 1
num_episodes = 1010
num_agents = 2

alpha = 0.5
decay_alpha = False
alpha_decay_rate = 0.999
tau = 1.0
low_tau = 0.1
decay_tau = True
tau_decay_rate = 0.99

## AC Start ##
alpha_theta = 0.00961 # low value when target is mixed policy like [0.25 0.75], but use higher rate e.g. 0.2 when target is pure policy like [0,1] ??
alpha_theta_decay = 1
alpha_q = 0.5
alpha_q_decay=0.999
## AC End ##

# arrays to store action reward values for action 1 after each episode so that these values maybe graphed
# these arrays will be used on the y-axis(vertical)
# action 1, would be stag in stag hunt for example, depending on payoff matrix setup. 
action_value_agent0_action1 = [] 
action_value_agent1_action1 = []

# for graph, just want to record episode number to use on x-axis(horizontal)
episodes = []
for i in range(0, num_episodes):
    episodes.append(i)

# arrays to store action values for action 1 after each episode so that these values maybe graphed
# these arrays will be used on the y-axis(vertical)
# action 1, would be stag in stag hunt for example, depending on payoff matrix setup. 
# for deterministic, values stored will be 0 or 1
actions1_agent0 = [] 
actions1_agent1 = []

actions2_agent0 = [] # required for graphs where showing 2 actions
showTwoActions=False # Set to True if showing graph with 2 actions for agent 0

# store probs on each episode from softmax - these values are included in a plot
exp_x_agent0=[]
exp_x_agent1=[]

useTarget = True  
budget = 10
adjust_by = 0.1 
useNash = 2 # 0, aim for highest reward. 1, Nash with max sum of payoffs. 2, Nash which is not max sum of payoffs. 
            # 3, if more than one Nash exists, give both players a different target but both represent Nash Equilibrium, 
            # but player 1 target results in a higher sum of rewards than player 2.
            # 4, same as 3 except player 2 gets the target with the higher sum of rewards

# some paramters to try out agent ignoring recourse at certain intervals
ignoreRecourse=False # if useSoftmax parameter used then this is redundant, so can remove it...
# Agent 0 will ignore recourse at intervals specified in this array
atIntervals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 

manSetRanges=False # if set to true the target will be manually set by the user, not based on useNash

# parameter to determine if softmax, recourse or both should be used for action selection
# 0 - softmax, no recourse
# 1 - softmax, recourse (redundant, get same results as if set to 2. Just have it here to test and show this.) 
# 2 - recourse, no softmax. Could be any other value apart from 2. Once it is not set to 0 or 1 or 3, then this is the default. 
# useSoftmax = 2, and = 1, should produce the same result. 
# 3 - softmax with a bit of recourse for a few episodes here and there.
useSoftmax = 3

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
# payoffs = [
#      [[1, -1], [-1, 1]],
#      [[-1, 1], [1, -1]]
# ]

# stag hunt NFG
payoffs = [
    [[5, 5], [0, 2]],
    [[2, 0], [1, 1]]
]

# rock paper scissors NFG
# payoffs = [
#     [[0, 0], [-1, 1], [1, -1]],
#     [[1, -1], [0, 0], [-1, 1]],
#     [[-1, 1], [1, -1], [0, 0]]
# ]

# prisoner dilemma NFG
# payoffs = [
#     [[3, 3], [0, 5]],
#    [[5, 0], [1, 1]]
# ]

# An asymmetric normal form game - Battle of the sexes - both want different things (e.g. one wants football, the other wants opera)
# payoffs = [
#     [[2, 1], [0, 0]],
#    [[0, 0], [1, 2]]
# ]

# Joint Venture (invest, not invest) NFG - Fully Cooperative Game
# payoffs = [
#     [[10, 10], [-5, 0]],
#    [[0, -5], [0, 0]]
# ]

# Battle of the sexes NFG - Coordination Game
# payoffs = [
#     [[2, 2], [0, 0], [0, 0]],
#     [[0, 0], [2, 2], [0, 0]],
#     [[0, 0], [0, 0], [2, 2]]
# ]

# Battle of the sexes NFG - Another version
# payoffs = [
#     [[3, 2], [0, 0], [0, 0]],
#     [[0, 0], [2, 3], [0, 0]],
#     [[0, 0], [0, 0], [1, 1]]
# ]

# Chicken Game NFG - Mixed Strategy Equilibrium
# payoffs = [
#     [[0, 0], [-1, 1], [-1, 2]],
#     [[1, -1], [-10, -10], [1, 3]],
#     [[2, -1], [3, 1], [-10, -10]]
# ]

# Some made up NFG - expect probs to be (0.5, 0.5, 0) for both row and col
# payoffs = [
#     [[2, 2], [0, 0], [0, 0]],
#     [[2, 0], [0, 2], [0, 0]],
#     [[0, 0], [0, 0], [0, 0]]
# ]

# Some made up NFG - expect probs to be (0.5, 0.5, 0) for row and (0, 0.5, 0.5) for col
# payoffs = [
#     [[2, 0], [0, 0], [0, 0]],
#     [[2, 0], [0, 2], [0, 0]],
#     [[0, 0], [0, 0], [0, 2]]
# ]

# A four action normal form game
# payoffs = [
#     [[3, 2], [1, 1], [0, 3], [2, 2]],
#     [[2, 3], [2, 2], [1, 4], [3, 1]],
#     [[4, 0], [3, 1], [2, 2], [0, 3]],
#     [[1, 1], [4, 2], [3, 0], [2, 2]]    
# ]

# A four action normal form game [page 12 of Natural Minimax Players in FourActionGame folder]
# payoffs = [
#     [[-1, 1], [1, -1], [1, -1], [-1, 1]],
#     [[1, -1], [-1, 1], [1, -1], [-1, 1]],
#     [[1, -1], [1, -1], [-1, 1], [-1, 1]],
#     [[-1, 1], [-1, 1], [-1, 1], [1, -1]]    
# ]

# A five action normal form game
# payoffs = [
#     [[3, 1], [0, 0], [2, 2], [1, 3], [4, 0]],
#     [[1, 2], [2, 1], [3, 3], [4, 0], [0, 4]],
#     [[4, 0], [3, 4], [2, 1], [1, 2], [0, 3]],
#     [[2, 3], [1, 0], [4, 4], [3, 1], [0, 2]],
#     [[0, 4], [2, 3], [1, 0], [4, 2], [3, 1]]    
# ]

num_actions = len(payoffs)

def single_play_payoff(row, col, payoffs, nashCalc):
    probs = [] 
    no_of_actions = len(row)
    row_payoffs = []
    col_payoffs = []
    i = 0
    payoff = 0
    total_payoff = 0

    # row
    k = 0
    while i < no_of_actions:
        j = 0
        while j < no_of_actions:
            payoff += (payoffs[i][j][k] * col[j])
            j += 1
        print("row payoff:", payoff)
        row_payoffs.append(payoff)
        total_payoff += (payoff * row[i])
        payoff = 0
        i += 1
    probs.append(total_payoff)

    # column
    total_payoff = payoff = j = 0
    k += 1
    while j < no_of_actions:
        i = 0
        while i < no_of_actions:
            payoff += (payoffs[i][j][k] * row[i])
            i += 1
        print("col payoff:", payoff)
        col_payoffs.append(payoff)
        total_payoff += (payoff * col[j])
        payoff = 0
        j += 1
    probs.append(total_payoff)
    if nashCalc:
        return probs
    else:
        return probs, row_payoffs, col_payoffs

def do_episode(agent_list, ranges, episode_number):
    probs_selected_actions = []
    a = 0 # which agent. 0 for agent 0, then 1 for agent 1.
    for agent in agent_list:
        prob_action = agent.select_action(a, ranges[a], episode_number) 
        probs_selected_actions.append(prob_action)
        a += 1
    
    print("SELECTED ACTIONS ARE:", probs_selected_actions)
    received_payoffs, row_payoffs, col_payoffs  = single_play_payoff(probs_selected_actions[0], probs_selected_actions[1], payoffs, False) # call this instead: 

    row_col_payoffs = []
    row_col_payoffs.append(row_payoffs)
    row_col_payoffs.append(col_payoffs)

    print("received_payoffs:", received_payoffs, "row_payoffs: ", row_payoffs, "col_payoffs:", col_payoffs)
    
    for a in range(len(agent_list)):
        # print("before update: agent", a, "action values",  agent_list[a].action_values)
        agent_list[a].update(row_col_payoffs[a])
        # print("after update: agent", a, "action values",  agent_list[a].action_values)

        # maybe change this code into a loop, could be used for more than two players. May need to use other list to store arrays actions1_agent0 etc...
        if a == 0:
            action_value_agent0_action1.append(agent_list[a].action_values[0])
            actions1_agent0.append(probs_selected_actions[a][0])
            actions2_agent0.append(probs_selected_actions[a][1])
        else: 
            action_value_agent1_action1.append(agent_list[a].action_values[0])
            actions1_agent1.append(probs_selected_actions[a][0])

def target(payoffs, useNash):
    agent0_probs = []
    agent1_probs = []

    def target_noNash():
        # Initialize variables to keep track of the maximum values and their indices
        max_value_row = float('-inf')  # negative infinity
        max_value_col = float('-inf')
        row_indices = []
        col_indices = []

        for i in range(len(payoffs)):
            for j in range(len(payoffs[i])):
                row_value = payoffs[i][j][0]  # Value for the row player
                col_value = payoffs[i][j][1]  # Value for the column player

                if row_value > max_value_row:
                    max_value_row = row_value
                    row_indices = [(i, j)]
                elif row_value == max_value_row:
                    row_indices.append((i, j))

                if col_value > max_value_col:
                    max_value_col = col_value
                    col_indices = [(i, j)]
                elif col_value == max_value_col:
                    col_indices.append((i, j))

        print(f"Payoff matrix: {payoffs}")
        # Print the results for row player
        print(f"The maximum value for the row player is {max_value_row}")
        print(f"Locations in the matrix for the row player (row index, column index): {row_indices}")

        # Print the results for column player
        print(f"The maximum value for the column player is {max_value_col}")
        print(f"Locations in the matrix for the column player (row index, column index): {col_indices}")

        print(f"Number of actions is: {len(payoffs)}")
        print(f"Number of actions for row to choose: {len(row_indices)}")
        print(f"Number of actions for column to choose: {len(col_indices)}")

        # for i in range(len(payoffs)):
        #     for j in range(len(row_indices)):

        for i in range(len(payoffs)):
            agent0_probs.append(0)
            agent1_probs.append(0)
        for i in range(len(row_indices)):
            agent0_probs[row_indices[i][0]] = 1 / len(row_indices)
        for i in range(len(col_indices)):
            agent1_probs[col_indices[i][1]] = 1 / len(col_indices)
        print(f"agent0_probs target: {agent0_probs}")
        print(f"agent1_probs target: {agent1_probs}")

    if useNash not in (1, 2, 3, 4):
        # Iterate through the matrix to find the maximum value for both players and their indices
        target_noNash()
    else:
        # need to calculate Nash
        row_player = []
        col_player = []

        for a in range(len(payoffs)):
            temp = []
            for b in range(len(payoffs[a])):
                temp.append(payoffs[a][b][0])
            row_player.append(temp)
            temp = []
            for b in range(len(payoffs[a])):
                temp.append(payoffs[a][b][1])
            col_player.append(temp)

        print(row_player, col_player)

        A = np.array(row_player)
        B = np.array(col_player)

        nfg = nash.Game(A, B)

        ne = list(nfg.vertex_enumeration())

        if len(ne) > 1: # more than one nash exists
            print("more than one nash exists", ne)
            agent0_one_nash = []
            agent1_one_nash = []
            max = float('-inf')
            for eq in ne:
                received_payoffs = single_play_payoff(eq[0], eq[1], payoffs, True)
                total = sum(received_payoffs)
                if total > max:
                    max = total
                    agent0_one_nash.clear()
                    agent0_one_nash.append(eq[0])
                    agent0_one_nash.append(eq[1])
                    agent1_one_nash.clear()
                    agent1_one_nash.append(eq[0])
                    agent1_one_nash.append(eq[1])
            # Have max, but need to select another if not max required
            if useNash == 2: # This code only runs if more than one nash exists - see parent if condition
                print("not max - pick one randomly which is not max")
                for eq in ne:
                    temp = []
                    temp.append(eq[0])
                    temp.append(eq[1])
                    if not np.array_equal(np.array(temp), np.array(agent0_one_nash)):
                        agent0_one_nash.clear()
                        agent0_one_nash.append(eq[0])
                        agent0_one_nash.append(eq[1])
                        agent1_one_nash.clear()
                        agent1_one_nash.append(eq[0])
                        agent1_one_nash.append(eq[1])  
                        break  # once a match is found which is not Nash with the maximum payoff, exit with that value
            
            if useNash == 3:  # This code only runs if more than one nash exists - see parent if condition
                # Holder - Incomplete
                print("player 1 gets target with bigger sum of rewards than player 2")
                # player 1 already has target which is Nash with the biggest sum of rewards compared to other targets with Nash - see code block from line 320
                # Just assign a target which results in Nash with a lesser sum of rewards to player 2
                for eq in ne:
                    temp = []
                    temp.append(eq[0])
                    temp.append(eq[1])
                    if not np.array_equal(np.array(temp), np.array(agent0_one_nash)): # if its not equal then it has lower sum of rewards
                        agent1_one_nash.clear()
                        agent1_one_nash.append(eq[0])
                        agent1_one_nash.append(eq[1])  
                        break  # once a match is found which is not Nash with the maximum payoff, exit with that value

            if useNash == 4: #  This code only runs if more than one nash exists - see parent if condition
                # Holder - Incomplete
                print("player 2 gets target with bigger sum of rewards than player 1")
                # player 2 already has target which is Nash with the biggest sum of rewards compared to other targets with Nash - see code block from line 320
                # Just assign a target which results in Nash with a lesser sum of rewards to player 1
                for eq in ne:
                    temp = []
                    temp.append(eq[0])
                    temp.append(eq[1])
                    if not np.array_equal(np.array(temp), np.array(agent0_one_nash)): # if its not equal then it has lower sum of rewards
                        agent0_one_nash.clear()
                        agent0_one_nash.append(eq[0])
                        agent0_one_nash.append(eq[1])  
                        break  # once a match is found which is not Nash with the maximum payoff, exit with that value

            # only accommodates two players at present but will accommodate two or more actions            
            i = 0
            j = 0
            while j < len(agent0_one_nash[i]):    
                agent0_probs.append(agent0_one_nash[i].tolist()[j])
                j += 1
            j = 0
            i += 1
            while j < len(agent1_one_nash[i]):    
                agent1_probs.append(agent1_one_nash[i].tolist()[j])
                j += 1
        else:
            print("only one or no nash exists")
            if len(ne) == 1:              
                # only accommodates two players at present but will accommodate two or more actions            
                i = 0
                j = 0
                while j < len(ne[0][i]):    
                    agent0_probs.append(ne[0][i].tolist()[j])
                    j += 1
                j = 0
                i += 1
                while j < len(ne[0][i]):    
                    agent1_probs.append(ne[0][i].tolist()[j])
                    j += 1
            else:
                # same code as used for useNash = 0. 
                # will always be at least one Nash. If not a pure strategy then a mixed one for Nash will exist.
                target_noNash()

    # scale action prob values to ensure they sum up to 1. Should do, but could be slightly off with rounding.
    total_sum = sum(agent0_probs)
    # Scale each element by dividing by total_sum
    scaled_values = [round(x / total_sum, 2) for x in agent0_probs]
    scaled_values[-1] = 1 - sum(scaled_values[:-1])
    agent0_probs = scaled_values
    total_sum = sum(agent1_probs)
    # Scale each element by dividing by total_sum
    scaled_values = [round(x / total_sum, 2) for x in agent1_probs]
    scaled_values[-1] = 1 - sum(scaled_values[:-1])
    agent1_probs = scaled_values

    # Calculate ranges for each agent. In most cases it will be the same for both agents - I assume so, may need to confirm this
    # Just calculating for one agent for now.

    #temp, need to comment out as just using to fix target action probability values for now.
    agent0_probs = [0.25, 0.75]
    agent1_probs = [0.25, 0.75]
    ranges_agent0, ranges_agent1 = setRanges(agent0_probs, agent1_probs)

    return agent0_probs, agent1_probs, ranges_agent0, ranges_agent1

def setRanges(agent0_probs, agent1_probs):
    print("Set ranges")
    print(f"probs for agent 0 are {agent0_probs} and length is {len(agent0_probs)}")
    ranges_agent0 = []
    for i in range(len(agent0_probs)):
        if agent0_probs[i] in (0, 1): # instead of 0, use 0.01. Had a highly unlikely occurence once where all probs choose a value of 0 randomly. This caused problems when dividing by 0, so making minimum value 0.01.
            ranges_agent0.append([0, 1])
        else:
            ranges_agent0.append([0, agent0_probs[i]])

    print(f"probs for agent 1 are {agent1_probs} and length is {len(agent1_probs)}")
    ranges_agent1 = []
    for i in range(len(agent1_probs)):
        if agent1_probs[i] in (0, 1):
            ranges_agent1.append([0, 1])
        else:
            ranges_agent1.append([0, agent1_probs[i]])

    return ranges_agent0, ranges_agent1

class Agent:
    def __init__(self, a, t, u, a_id):
        self.alpha = a
        self.tau = t
        self.action_values = [0 for a in range(num_actions)]
        
        ## AC Start ##
        print("self.u :::::::::::::::::::::: ", u)
        self.u = u
        self.grad = jit(grad(self.objective_function))
        self.alpha_theta = alpha_theta # 0.01
        self.alpha_theta_decay = alpha_theta_decay # 1
        self.alpha_q = alpha_q # 0.01
        self.alpha_q_decay = alpha_q_decay # 1
        #self.theta = np.zeros(num_actions)
        # May have to initialise theta based on taget policy - mulitply the values for target by 10, or some value to give theta.
        # Generally target policy for each agent will be the same, but in some instances maybe different, but can still probably multiply by 10 for both agents, or by some other value which is the same. May need to try and figure out how to get the optimal initial theta values..
        #self.theta = np.array([0.0, 10.0]) # want to encourage policy which is more inclined to 2nd action - if target policy is e.g. [0, 1]
        #self.theta = np.array([10.0, 0.0]) # want to encourage policy which is more inclined to 1st action - if target policy is e.g. [1, 0]
        self.theta = np.array([0.0, 4.45]) # want to encourage policy which is slightly more inclined to 2nd action - if target policy is e.g. [0.25, 0.75]
        self.policy = softmax_strategy(self.theta)
        self.q_values_ac = np.zeros((num_actions, 1))
        self.a_id = a_id
        ## AC End ##

    ## AC Start ##    
    def objective_function(self, theta, q_values_ac):
        """The objective function for the agent. This is the SER criterion.

        Args:
            theta (ndarray): The policy parameters.
            q_values_ac (ndarray): The expected returns for the actions.

        Returns:
            float: The utility from the current policy and Q-values.

        """
        policy = softmax(theta)
        expected_returns = jnp.matmul(policy, q_values_ac)
        #utility = self.u[self.a_id](expected_returns)
        utility = self.u[0](expected_returns) # always 0, each agent on there turn is designated as player 1 for utility purpose
        return utility
    ## AC End ##

    def softmax_select_action(self, q_values, tau, a):
        print("ITS SOFTMAX EXPLORE:", q_values, "TAU is ", tau)

        # modify q value here, but scale of boost - need to be careful - maybe use a percentage, needs to be close to actual.
        exp_x = np.exp(q_values/tau)
        exp_x /= np.sum(exp_x) # modify the choice instead. e.g update heads to 0.8 and tails to 0.2
        print("SOFTMAX for agent ",a,"::::::EXP::::::",exp_x,"EXP 0:", exp_x[0], "EXP 1:", exp_x[1])

        if a == 0:
            exp_x_agent0.append(exp_x[0])
        else:
            exp_x_agent1.append(exp_x[0])
        
        return np.random.choice(len(q_values), p=exp_x)

    def select_action(self, a, ranges, episode_number):   # will rename to set_action.     
        q_values = np.array(self.action_values)
        print("SOFTMAX: self.action_values:", q_values)
        print("EPISODE NUMBER:", episode_number)
        # may not be required as use each q value - see line 487 - softmax_action < l - 1:
        # will select all the q values. All softmax_action gives is where to start - the first or second or third etc q-value
        # could just set softmax_action = 0 and it should work the same?
        #softmax_action = self.softmax_select_action(q_values, self.tau, a) 

        # Ignore softmax choice and just go with first action. 
        # A value picked from a range will be assigned to the first action, 
        # then move onto the second action, then the third (if one in the nfg being learned) etc.. 
        # All actions will be updated with a value selected from a range.
        #softmax_action = 0 

        if useSoftmax == 0:
            softmax_action = self.softmax_select_action(q_values, self.tau, a)
            # need a scaled value array which has length == no of actions. initialise all
            # values to 0, and just update 1 with value 1 based on softmax selection
            # then return scaled value array. that gets us out of this function.
            scaled_values = np.zeros(len(q_values))
            scaled_values[softmax_action] = 1
            print("scaled_values", scaled_values, "softmax_action", softmax_action)
            return scaled_values
        elif useSoftmax == 1:
            softmax_action = self.softmax_select_action(q_values, self.tau, a)
        elif useSoftmax == 3:
            # sr = 0
            # r_ranges = [(90, 100), (190, 200), (290, 300), (390, 400), (490, 500), (590, 600), (690, 700), (790, 800)]
            # # Check if the number is within any of the ranges
            # for start, end in r_ranges:
            #   if start <= episode_number <= end:
            #       sr = 1
            if episode_number % 100 >= 0 and episode_number % 100 < 10:
               #Perform recourse for 10 iterations starting from 100, 200, 300, etc.
            #if sr == 1: 
               softmax_action = self.softmax_select_action(q_values, self.tau, a) # just need this for producing the final graph, ignore action returned - see next line
               softmax_action = 0
               print("EPISODE USES RECOURSE")
            else: # same as if useSoftmax = 0. Maybe put this code into a function as its used twice
                softmax_action = self.softmax_select_action(q_values, self.tau, a)
                # need a scaled value array which has length == no of actions. initialise all
                # values to 0, and just update 1 with value 1 based on softmax selection
                # then return scaled value array. that gets us out of this function.
                scaled_values = np.zeros(len(q_values))
                scaled_values[softmax_action] = 1
                print("scaled_values", scaled_values, "softmax_action", softmax_action)
                return scaled_values
        else:
            softmax_action = self.softmax_select_action(q_values, self.tau, a) # just need this for producing the final graph, ignore action returned - see next line
            softmax_action = 0 # not using softmax

        l = len(payoffs)
        action_probs = []
        for i in range(l):
            action_probs.append(0)

        for i in range(l):
            if useTarget:
                action_probs[softmax_action] = round(random.uniform(ranges[softmax_action][0], ranges[softmax_action][1]), 2)
            else:
                # otherwise choose random values from 0 to 1 for each action. Dont use ranges list
                action_probs[softmax_action] = round(random.uniform(0, 1), 2) 
            if softmax_action < l - 1:
                softmax_action += 1
            else:
                softmax_action = 0

        print(action_probs, sum(action_probs))

        # looping though values to make sure none of the action_probs are 0. This will cause a problem with scaling.
        # It is highly unlikely any will be, but in one test the random numbers for each value produced 0, so it could
        # happen. Rounding to 2 decimal places don't exist. If any of the values are 0, then update them to 0.01
        for a in range(len(action_probs)):
            if action_probs[a] == 0.0:
                action_probs[a] = 0.01
        
        # Calculate the sum of the array
        total_sum = sum(action_probs)

        # Scale each element by dividing by total_sum
        scaled_values = [round(x / total_sum, 2) for x in action_probs]
        scaled_values[-1] = 1 - sum(scaled_values[:-1])
        # Print the scaled values
        print(scaled_values, sum(scaled_values))

        return scaled_values
    
    def update(self, payoff):
        for i in range(len(payoff)):
            print("self.action_values[i]", self.action_values[i], "alpha", self.alpha, "payoff[i]", payoff[i])
            self.action_values[i] += self.alpha * (payoff[i] - self.action_values[i])  
            print("action value", i, "", self.action_values[i])
            self.update_q_table(i, payoff[i])
            self.update_ac() ## AC ##

    ## AC Start ##
    def update_ac(self):
        self.theta += self.alpha_theta * self.grad(self.theta, self.q_values_ac)
        self.policy = softmax_strategy(self.theta)
        print("policy:", self.policy, "agent:", self.a_id, "theta", self.theta, "self.q_values_ac", self.q_values_ac)
        #self.update_parameters()

    def update_parameters(self):
        """Update the hyperparameters. Decays the learning rate for the Q-values and policy parameters."""
        self.alpha_q *= self.alpha_q_decay
        self.alpha_theta *= self.alpha_theta_decay

    def update_q_table(self, action, reward):
        """Update the Q-table based on the chosen actions and the obtained reward.

        Args:
            action (int): The action chosen by this agent.
            reward (float): The reward obtained by this agent.

        Returns:

        """
        self.q_values_ac[action] += self.alpha_q * (reward - self.q_values_ac[action])
        
    ## AC End ##

def main():
    ranges = []
    ranges_agent0 = []
    ranges_agent1 = []
    if useTarget:
        agent0_probs, agent1_probs, ranges_agent0, ranges_agent1 = target(payoffs, useNash) # useNash 0, 1, 2, 3 -- see near head of program for value meanings
        print("######## Agent 0 Probs ###########", agent0_probs)
        print("######## Agent 1 Probs ###########", agent1_probs)
    else:
        for i in range(len(payoffs)):
            ranges_agent0.append([0, 1])
            ranges_agent1.append([0, 1])   
    ranges.append(ranges_agent0)
    ranges.append(ranges_agent1)

    print("######## RANGES ###########", ranges)
   

    if manSetRanges: # manually change the target probs for an agent. E.g. change target on prisoner to cooperate([1.0, 0.0]) for each agent instead of defect ([0.0, 1.0]).
        agent0_probs = [1.0, 0.0]
        agent1_probs = [1.0, 0.0]

    for run in range(num_runs):
        print("----- Beginning run", run, "-----")
        
        agent_list = []
        for agent in range(num_agents):
            agent_list.append(Agent(alpha, tau, u_tpl, agent))
        
        # for range adjustment on action selection
        softmax_episode = 0

        # for range adjustment if recourse is getting ignored on an interval
        # keeps track if ignoring recourse for more than 1 interval in a row
        consecutiveCount = 0
        
        for episode in range(num_episodes):
            softmax_episode += 1
            print("     *** Episode", episode, "***")   
            
            # on episodes 330, 500, 670 add noise by corrupting ranges.
            if softmax_episode in [1330000, 15000000, 16700000]: # set to values greater than number of episodes if you don't want noise added
                print("corrupting ranges")
                for i in range(len(agent0_probs)): 
                    ranges_agent0[i][1] = 1
                    ranges_agent0[i][0] = 0
                for i in range(len(agent1_probs)): 
                    ranges_agent1[i][1] = 1
                    ranges_agent1[i][0] = 0
                ranges.clear()
                ranges.append(ranges_agent0)
                ranges.append(ranges_agent1)       
            
            do_episode(agent_list, ranges, softmax_episode)
            
            if decay_alpha:
                for agent in agent_list:
                    agent.alpha = agent.alpha * alpha_decay_rate
                    print("Alpha for agent is", agent.alpha)
            if decay_tau:
                for agent in agent_list:
                    agent.tau = max(low_tau, agent.tau * tau_decay_rate)
            if useTarget: # adjust ranges if using recourse to push towards target - adjust_by value determines by how. If left at 0, then it is the same as not using recourse            
                if (softmax_episode % (num_episodes/budget)) == 0:

                    # reset the ranges to original values
                    # useful if during learning, noise impacts ranges, thus learning
                    # if using recourse, then reset the values. 
                    # This will enable agents to overcome any issues during learning and lesson the impact of them.

                    # in summary, as part of recourse, look at target, reset the ranges, then adjust.
                    ranges_agent0, ranges_agent1 = setRanges(agent0_probs, agent1_probs)
                    
                    # require this to determine what to adjust the ranges to, is it the first adjustment, second adjustment etc ..
                    factor = softmax_episode / (num_episodes / budget)
                    # factor = 1

                    # checking if agent 0 will ignore recourse at certain intervals.
                    # Using factor_agent0 to determine range for agent 0, instead of factor, 
                    # incase agent 0 is configured for ignoring recourse. 
                    if(((softmax_episode / (num_episodes/budget)) in atIntervals) and ignoreRecourse):
                        consecutiveCount += 1
                    else:
                        consecutiveCount = 0
                    
                    factor_agent0 = factor - consecutiveCount

                    for i in range(len(agent0_probs)): # have two for loops. one for each agent, but what if I have more than 2 agent - need to modify, use lists and loops instead
                        if agent0_probs[i] == 0:
                            ranges_agent0[i][1] = ranges_agent0[i][1] - (adjust_by * factor_agent0)
                        elif agent0_probs[i] == 1:
                            ranges_agent0[i][0] = ranges_agent0[i][0] + (adjust_by * factor_agent0)
                        else:
                            ranges_agent0[i][0] = ranges_agent0[i][0] + ((adjust_by * factor_agent0) * agent0_probs[i]) # agent0_probs[0] not 1 or 0, so adjust according to its value 
                    
                    for i in range(len(agent1_probs)):
                        if agent1_probs[i] == 0:
                            ranges_agent1[i][1] = ranges_agent1[i][1] - (adjust_by * factor)
                        elif agent1_probs[i] == 1:
                            ranges_agent1[i][0] = ranges_agent1[i][0] + (adjust_by * factor)
                        else:
                            ranges_agent1[i][0] = ranges_agent1[i][0] + ((adjust_by * factor) * agent1_probs[i])
                   
                    ranges.clear()
                    ranges.append(ranges_agent0)
                    ranges.append(ranges_agent1)
                    # comment out below. Just have for now to manually set ranges - will use same range wherever it is in learning cycle.
                    #ranges.clear()
                    #ranges.append([[0.23, 0.26], [0.73, 0.76]])
                    #ranges.append([[0.23, 0.26], [0.73, 0.76]])
                    
        print(consecutiveCount)                
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

        ## AC Start ##
        print("AC agent", 0, "policy", agent_list[0].policy)
        print("AC agent", 1, "policy", agent_list[1].policy)
        
        # q_values_ac
        print("AC agent", 0, "q_values_ac", agent_list[0].q_values_ac)
        print("AC agent", 1, "q_values_ac", agent_list[1].q_values_ac)
        
        ## AC End ##

        if useTarget:
            print(f"agent0_probs {agent0_probs}, agent1_probs {agent1_probs}, ranges_agent0 {ranges_agent0}, ranges_agent1 {ranges_agent1}")

        #draw_graph(episodes, action_value_agent0_action1, action_value_agent1_action1, "Q Values per episode", "Q Values")
        #draw_graph(episodes, actions1_agent0, actions1_agent1, "Actions per episode", "Probability of action 0")
        #if useSoftmax != 0:
            #draw_graph(episodes, exp_x_agent0, exp_x_agent1, "Actions per episode: Softmax", "Probability of action 0")

        # other plots just show for first action over number of episodes
        # if want to show values for two of the available actions for one player, 
        # then set showTwoActions property to True.
        if showTwoActions: 
            split  = len(actions1_agent0) // 2
            
            plt.scatter(actions1_agent0[:split], actions2_agent0[:split], color='blue', s=7)
            plt.scatter(actions1_agent0[split:], actions2_agent0[split:], color='green', s=7)
            #plt.scatter(actions1_agent0, actions2_agent0, s=5, color='blue')
            plt.xlabel('Probability of action 0 (Paper)')
            plt.ylabel('Probability of action 1 (Scissors)')
            plt.title('Action Probabilities for one player')
            plt.show()

def draw_graph(x_axis, y_axis1, y_axis2, title, y_label):
    # Agent 0
    x_values = x_axis 
    y_values = y_axis1
    plt.plot(x_values, y_values, marker='', color="orange", label="Agent 0")
    # Agent 1
    x_values = x_axis  
    y_values = y_axis2 
    plt.plot(x_values, y_values, marker='', color="blue", label="Agent 1")

    plt.xlabel('Episodes')
    plt.ylabel(y_label)
    plt.title(title)
    x = 0.9
    y = 0.2
    plt.legend(loc='upper right', bbox_to_anchor=(x, y))
    plt.show()

if __name__ == "__main__":
    main()