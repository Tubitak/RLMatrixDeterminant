import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Determinant_v0(gym.Env):
    """Determinant
    The goal of Determinant is to find a non-invertible matrix

    After each step the agent receives an observation of:
    r - determinant of matrix

    The rewards is calculated as:
    (det(new_matrix) - det(old_matrix)) ** 2

    TODO: Vidi li celu matricu? Zasad izgleda da ne. Takodje, popraviti sve kao u v1. Za sad nije dobro.
    """
    def __init__(self):

        self.target_determinant = 0.0

        self.n = 6

        self.action_space = spaces.Box(low=np.array([0]), high=np.array([self.n**2]))
        self.observation_space = spaces.Box(low=np.array([-100]), high=np.array([100]))

        self.matrix = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
        self.guess_count = 0
        self.guess_max = 200
        self.observation = [np.linalg.det(self.matrix)]

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        action = int(action[0])  # korektno?

        value_on_action = self.matrix.reshape(self.n**2)[action]
        if value_on_action == 0.:
            new_matrix = self.matrix.reshape(self.n**2)
            new_matrix[action] = 1.
            self.matrix = new_matrix.reshape([self.n, self.n])
        if value_on_action == 1.:
            new_matrix = self.matrix.reshape(self.n**2)
            new_matrix[action] = 0.
            self.matrix = new_matrix.reshape([self.n, self.n])

        reward = - (self.observation[0] - np.linalg.det(self.matrix)) ** 2
        self.observation = [np.linalg.det(self.matrix)]

        self.guess_count += 1
        done = self.guess_count >= self.guess_max

        return self.observation, reward, done, {"matrix": self.matrix, "guesses": self.guess_count}

    def _reset(self):
        self.matrix = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
        self.guess_count = 0
        self.observation = [1.]
        return self.observation


class Determinant_v1(gym.Env):
        """Determinant
        The goal of Determinant is to find a non-invertible matrix

        After each step the agent receives an observation of:
        m - matrix

        The rewards is calculated as:
        (det(new_matrix) - det(old_matrix)) ** 2

        TODO: Implementirati da gadja tacno target_determinant
        """

        def __init__(self):

            self.target_determinant = 1.0

            self.n = 6

            self.action_space = spaces.Box(low=np.array([0]), high=np.array([self.n ** 2]))
            self.observation_space = spaces.Box(low=0., high=1., shape=(self.n * self.n))

            self.matrix = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
            self.guess_count = 0
            self.guess_max = 200
            self.observation = self.matrix.reshape(self.n**2)  # A view of self.matrix, not a new array

            self._seed()
            self._reset()

        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

        def _step(self, action):
            # TODO: srediti ove crne oblike

            assert self.action_space.contains(action)

            action = int(action[0])  # korektno?

            value_on_action = self.matrix.reshape(self.n ** 2)[action]
            old_matrix = np.array(self.matrix)

            if value_on_action == 0.:
                new_matrix = self.matrix.reshape(self.n ** 2)
                new_matrix[action] = 1.
                self.matrix = new_matrix.reshape([self.n, self.n])
            if value_on_action == 1.:
                new_matrix = self.matrix.reshape(self.n ** 2)
                new_matrix[action] = 0.
                self.matrix = new_matrix.reshape([self.n, self.n])


            EPSILON = 0.0001
            reward = 0
            if (self.target_determinant - np.linalg.det(self.matrix)) ** 2 >= EPSILON:
                reward = -1
            if (self.target_determinant - np.linalg.det(self.matrix)) ** 2 < EPSILON:
                reward = 20

            self.observation = self.matrix.reshape(self.n ** 2)

            self.guess_count += 1
            done = self.guess_count >= self.guess_max or reward > 0

            #if done:
            #    print(self.matrix, np.linalg.det(self.matrix))

            return self.observation, reward, done, {"matrix": self.matrix, "guesses": self.guess_count}

        def _reset(self):
            self.matrix = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
            self.guess_count = 0
            self.observation = self.matrix.reshape(self.n**2)
            return self.observation
