from rlberry.experiment import load_experiment_results
from rlberry.manager.evaluation import plot_writer_data
from rlberry.agents.dynprog import ValueIterationAgent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 10,7
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 14

# ------------------------------------------
# RandQL vs OptQL
# ------------------------------------------
EXPERIMENT_NAME = 'randql_vs_baseline'
RESULTS_PATH = 'results'


# Get list of managers and update names
PLOT_TITLES = {
    'optql': 'OptQL',
    'randql': 'Randomized QL',
    'ucbvi': 'UCBVI',
    'rlsvi': 'RLSVI',
    'psrl': 'PSRL',
    'greedy_ucbvi': 'Greedy UCBVI'
}
output_data = load_experiment_results(RESULTS_PATH, EXPERIMENT_NAME)
_manager_list = list(output_data['manager'].values())
manager_list = []
agents_list = []
# Sort by names
_manager_list = sorted(_manager_list, key=lambda x: x.agent_name)

for manager in _manager_list:
        if manager.agent_name in PLOT_TITLES:
            manager.agent_name = PLOT_TITLES[manager.agent_name]
            manager_list.append(manager)
            print(manager.agent_name)
            print("n agents = ", len(manager.get_agent_instances()))
            agents_list.append(manager.get_agent_instances()[0])
            del manager.agent_handlers

# Get value of optimal agent
env = manager_list[0].train_env
horizon = agents_list[0].horizon
vi_agent = ValueIterationAgent(env, gamma=1.0, horizon=horizon)
vi_agent.fit(budget=1000)
v_star = vi_agent.V[0,vi_agent.env.reset()]

def compute_regret(episode_rewards):
   return np.cumsum(v_star - episode_rewards)
res = plot_writer_data(manager_list, tag="episode_rewards", show = False, preprocess_func=compute_regret, title=' ')
plt.ylabel('regret')
plt.xlabel('episode')
plt.grid()
plt.savefig('{}.pdf'.format(EXPERIMENT_NAME))