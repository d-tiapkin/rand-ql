from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom

def constructor(nrooms, room_size, success_probability):
    env = NRoom( nrooms=nrooms,
                 room_size=room_size,
                 success_probability=success_probability)
    env.terminal_states = ()
    return env