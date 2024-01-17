import numpy as np

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy
from rlberry.exploration_tools.discrete_counter import DiscreteCounter

import rlberry

logger = rlberry.logger


class RandomizedQLAgent(AgentWithSimplePolicy):
    """
    Randomized Q-Learning
    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    gamma : double, default: 1.0
        Discount factor in [0, 1].
    horizon : int
        Horizon of the objective function.
    bootstrap_samples: int, default: 10
        Number of bootstrap resamples
    prior_transitions: double, default: 1.0
        Number of prior transitions
    kappa: double, default: 1.0
        Posterior inflation coefficient
    scheduled: bool, default: False
        Type of updates of acting Q-function
    sampling: bool, default: False
        If true sample at random a Q-value 
        for the ensemble to act.
    References
    ----------
    None
    """

    name = "RandQL"

    def __init__(
        self,
        env,
        gamma=1.0,
        horizon=100,
        bootstrap_samples=10,
        prior_transitions=None,
        kappa=1.0,
        scheduled=False,
        sampling=False,
        **kwargs
    ):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.horizon = horizon
        self.bootstrap_samples = bootstrap_samples
        self.prior_transitions = prior_transitions
        if prior_transitions is None:
            self.prior_transitions = 1.0 /self.env.observation_space.n
        self.kappa = kappa
        self.scheduled = scheduled
        self.sampling = sampling

        self.steps_next_epoch = 1.0#self.horizon
        self.steps_increase = 1. + 1.0/(self.horizon)

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning(
                "{}: Reward range is  zero or infinity. ".format(self.name)
                + "Setting it to 1."
            )
            r_range = 1.0

        self.v_max = np.zeros(self.horizon)
        self.v_max[-1] = r_range
        for hh in reversed(range(self.horizon - 1)):
            self.v_max[hh] = r_range + self.gamma * self.v_max[hh + 1]

        # initialize
        self.reset()

    def reset(self, **kwargs):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n
        J = self.bootstrap_samples

        # (s, a) visit counter
        self.N_sa = np.zeros((H, S, A))
        self.N_sa_tilde = np.ones((H,S,A)) * self.prior_transitions * self.kappa

        # Value functions
        self.V = np.ones((H + 1, S))
        self.V[H, :] = 0
        self.Q = np.ones((H, S, A))
        self.Q_bar = np.ones((H, S, A))
        self.Q_tilde = np.ones((J, H, S, A))
        for hh in range(self.horizon):
            self.V[hh, :] *= self.horizon - hh
            self.Q[hh, :, :] *= self.horizon - hh
            self.Q_bar[hh, :, :] *= self.horizon - hh
            self.Q_tilde[:, hh, :, :] *= self.horizon - hh
        # ep counter
        self.episode = 0

        # useful object to compute total number of visited states & entropy of visited states
        self.counter = DiscreteCounter(
            self.env.observation_space, self.env.action_space
        )

    def policy(self, observation):
        """Recommended policy."""
        state = observation
        return self.Q_bar[0, state, :].argmax()

    def _get_action(self, state, hh=0, sampling_idx=None):
        """Sampling policy."""
        if sampling_idx is None:
            return self.Q_bar[hh, state, :].argmax()
        return self.Q_tilde[sampling_idx, hh, state, :].argmax()

    def _update(self, state, action, next_state, reward, hh):
        self.N_sa_tilde[hh, state, action] += 1
        nn = self.N_sa_tilde[hh, state, action]

        #Compute target 
        target = reward + self.gamma * self.V[hh+1, next_state]
        if not self.scheduled:
            # Mix with prior transition to not forget too fast 
            alpha_prior = nn
            beta_prior = self.prior_transitions
            weights_prior = self.rng.beta(
                alpha_prior, 
                beta_prior,
                size=self.bootstrap_samples
                )
            target = weights_prior * (reward + self.gamma * self.V[hh+1, next_state])
            # If sampling we use the corresponding Q-value to update the ensemble 
            if self.sampling:
                next_v = 0.0
                if hh < (self.horizon-1):
                    next_v = self.Q_tilde[:, hh+1, next_state, :].max(axis=-1)
                target = weights_prior * (
                    reward + self.gamma * next_v
                    )
            target += (1-weights_prior) * self.v_max[hh]


        #Get learning rate 
        if not self.scheduled:
            alpha = (self.horizon + 1.0) / self.kappa
        else:
            alpha = 1.0 / self.kappa
        beta = nn / self.kappa
        weights = self.rng.beta(alpha, beta, size=self.bootstrap_samples)
        
        # Update ensemble of Q-values
        self.Q_tilde[:, hh, state, action] = (1 - weights) * self.Q_tilde[:, hh, state, action] + weights * target

        if not self.scheduled:
            self.Q_bar[hh, state, action] = min(self.Q_bar[hh, state, action], self.Q_tilde[:, hh, state, action].max())
            self.V[hh, state] = min(self.v_max[hh], self.Q_bar[hh, state, :].max())

        # make scheduled update
        if self.scheduled and nn >= self.steps_next_epoch:
            #logger.info('New epoch for state s = {}, a = {}, h = {} at episode {}'.format(state, action, hh, self.episode))
            self.Q_bar[hh, state, action] =  min(self.Q_bar[hh, state, action], self.Q_tilde[:, hh, state, action].max())
            self.V[hh, state] = min(self.v_max[hh], self.Q_bar[hh, state, :].max())

            #w = np.random.beta(self.prior_transitions/2, self.prior_transitions/2)
            self.Q_tilde[:, hh, state, action] = self.v_max[hh]
            self.N_sa_tilde[hh, state, action] = self.prior_transitions * self.kappa
            self.steps_next_epoch *= self.steps_next_epoch  

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        #Sample an index for the episode if sampling is true
        sampling_idx = None
        if self.sampling:
            sampling_idx = self.rng.choice(self.bootstrap_samples)
        for hh in range(self.horizon):
            action = self._get_action(state, hh, sampling_idx)
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward  # used for logging only

            self.counter.update(state, action)

            self._update(state, action, next_state, reward, hh)

            state = next_state
            if done:
                break

        # update info
        self.episode += 1

        # writer
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)
            self.writer.add_scalar(
                "n_visited_states", self.counter.get_n_visited_states(), self.episode
            )

        # return sum of rewards collected in the episode
        return episode_rewards

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.
        Parameters
        ----------
        budget: int
            number of episodes. Each episode runs for self.horizon unless it
            enconters a terminal state in which case it stops early.
        """
        del kwargs
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
            self._run_episode()
            count += 1
