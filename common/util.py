import numpy as np
from gym.spaces import Box, Discrete

def space_n_to_shape_n(space_n):
    return np.array([space_to_shape(space) for space in space_n])

def space_to_shape(space):
    if type(space) is Box:
        return space.shape
    if type(space) is Discrete:
        return [space.n]
