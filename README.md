<h1>Recourse</h1>

The code in this repository demonstrates a use for recourse as part of reinforcement learning. The code exists in nfg_learning_recourse_nash_noise.py

<h2>Parameters</h2>

<h3>There are a number of key parameters.</h3>

The <b>payoffs</b> parameter. This is used for setting the rewards for a two player normal form game. A number of examples are included in the code and commented out. Comment and uncomment depending on which normal form game a strategy needs to be developed for. If other normal form game required than those available in the code, just ensure the rewards for the game are assigned to the payoffs parameter in a similar structure to the current examples in the code.

The <b>useTarget</b> parameter. Values for this are True or False. If set to True, this indicates recourse will be used as part of learning. A target for a strategy will be set and the learners will be encouraged towards this target while learning. If set to false, then reinforcement learning will occur without any help from recourse.

The <b>budget</b> parameter. This is used to indicate how many times during learning recourse will be used by the learners to seek guidance on its journey towards learning a strategy. 

The <b>adjust_by parameter</b>. This parameter is used by recourse for adjusting ranges for selecting action probability values. Each time recourse is used during learning, as determined by the budget, the adjust_to parameter indicates by how much to adjust the ranges for selection by to encourage the learner towards the target.

The <b>useNash</b> parameter. This parameter is used along with the useTarget parameter. When setting a target, is a target which results in nash equilibria required. If more than one nash equilibria exists for the normal form game which one to select. Select the one which produces a higher reward than the others, or select one which doesn't. Set useNash to 1, if nash equilibria with the highest reward is required, or set to 2 if not. If useNash is set to a value of 0, then select a target which produces the highest payoff for each agent, regardless if it happens to result in nash equilibria or not.

<h3>Other parameters of note:</h3> 

<b>num_runs</b> - which indicate how many times to run a set of learning episodes. Typically just set to 1.  
<b>num_episodes</b> - Number of learning episodes to run.
<b>num_agents</b> - Number of players for the normal form game. A fixed value of 2 is required for this.
<b>alpha, decay_alpha, alpha_decay_rate</b> - Parameters used for learning rate, and whether to decay and by what over a number to reduce it so that learner is relying more on past rewards than recent rewards by the end of learning, so that learning is slowed down as a strategy is converged on.  
<b>tau, low_tau, decay_tau, tau_decay_rate</b> - Parameters used for softmax. If decaying tau overtime can use a decay rate. The low_tau parameter ensures the tau value has a minimum value so it can't reduce any further. With softmax, tau is used for determine the balance between exploiting existing knowledge or exploring for new knowledge. A high tau value encourages exploring, while a low value encourages exploiting. Typically to begin with it is good to explore but overtime during the learning process as it is coming to an end and a strategy is being settled upon, then exploit is the preferred approach.
