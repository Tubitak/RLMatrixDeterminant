import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

EPSILON = 1e-10


def uporedi(a, b):
    if abs(a - b) < EPSILON:
        return 0
    if a > b:
        return 1
    if a < b:
        return -1


def uporedi_nizove(a, b):
    for i in range(len(a)):
        if uporedi(a[i], b[i]) == 1:
            return 1
        if uporedi(a[i], b[i]) == -1:
            return -1
    return 0


def sort_lex(a):
    length = len(a) - 1
    sorted = False
    while not sorted:
        sorted = True
        for i in range(length):
            if uporedi_nizove(a[i], a[i + 1]) == 1:
                sorted = False
                tmp = a[i + 1].copy()
                a[i + 1] = a[i].copy()
                a[i] = tmp.copy()
    return a


import networkx as nx


def spektar(g):
    return np.sort(nx.adjacency_spectrum(g).real)


def razlika_spektara(g1, g2):
    return np.linalg.norm(spektar(g1) - spektar(g2), 2).sum()


def razlika_mini_spektara(g1, g2):
    spektri1 = []
    spektri2 = []
    for i in g1.nodes():
        g = g1.copy()
        g.remove_node(i)
        spektri1.append(spektar(g))
    for i in g2.nodes():
        g = g2.copy()
        g.remove_node(i)
        spektri2.append(spektar(g))
    return np.linalg.norm((np.array(sort_lex(spektri1)) - np.array(sort_lex(spektri2))).flatten(), 2)


class Determinant_v0(gym.Env):
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

        self.n = 10

        self.action_space = spaces.Discrete(self.n ** 2)
        self.observation_space = spaces.Box(low=0., high=1., shape=(self.n * self.n))

        self.matrix = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
        self.guess_count = 0
        self.guess_max = 200
        self.observation = self.matrix.reshape(self.n ** 2)  # A view of self.matrix, not a new array

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # TODO: srediti ove crne oblike

        assert self.action_space.contains(action)

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
            reward = 1

        self.observation = self.matrix.reshape(self.n ** 2)

        self.guess_count += 1
        done = self.guess_count >= self.guess_max or reward > 0

        # if done:
        #    print(self.matrix, np.linalg.det(self.matrix))

        return self.observation, reward, done, {"matrix": self.matrix, "guesses": self.guess_count}

    def _reset(self):
        self.matrix = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
        self.guess_count = 0
        self.observation = self.matrix.reshape(self.n ** 2)
        return self.observation


class Determinant_v1(gym.Env):
    """Hrabro na problem mini-spektara!
    #TODO: Mozda sve iste matrice na pocetku, da mu bude lakse. Mozda je brze da ih sacuvas kao grafove, pa da ih menjas nego da pravis svaki put.
    """

    def __init__(self):

        self.target_determinant = 1.0

        self.n = 15

        self.action_space = spaces.Discrete(2 * self.n ** 2)
        self.observation_space = spaces.Box(low=0., high=1., shape=(2 * self.n * self.n))

        m1 = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
        m2 = m1.copy()
        self.matrix1 = np.tril(m1) + np.tril(m1, -1).T
        self.matrix2 = np.tril(m2) + np.tril(m2, -1).T
        self.matrix = np.append(self.matrix1.flatten(), self.matrix2.flatten())

        self.guess_count = 0
        self.guess_max = 300
        self.observation = self.matrix  # A view of self.matrix, not a new array

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # TODO: srediti ove crne oblike

        assert self.action_space.contains(action)

        value_on_action = self.matrix[action]

        if action < self.n ** 2:
            red = action // self.n
            kolona = action % self.n
            if value_on_action == 0.:
                self.matrix1[red, kolona] = 1.
                self.matrix1[kolona, red] = 1.
            if value_on_action == 1.:
                self.matrix1[red, kolona] = 0.
                self.matrix1[kolona, red] = 0.
        else:
            action = action - self.n ** 2
            red = action // self.n
            kolona = action % self.n
            if value_on_action == 0.:
                self.matrix2[red, kolona] = 1.
                self.matrix2[kolona, red] = 1.
            if value_on_action == 1.:
                self.matrix2[red, kolona] = 0.
                self.matrix2[kolona, red] = 0.

        self.matrix = np.append(self.matrix1.flatten(), self.matrix2.flatten())

        reward = 0

        g1 = nx.from_numpy_matrix(self.matrix1)
        g2 = nx.from_numpy_matrix(self.matrix2)

        d_spektri = razlika_spektara(g1, g2)
        d_mini_spektri = razlika_mini_spektara(g1, g2)

        reward = 1 - d_mini_spektri
        if d_spektri < EPSILON:
            reward = -1000

        self.observation = self.matrix

        self.guess_count += 1
        done = self.guess_count >= self.guess_max or (reward >= 1 - EPSILON and d_spektri >= EPSILON)

        if reward >= 1 - EPSILON and d_spektri >= EPSILON:
            print(d_spektri, d_mini_spektri)
            print(self.matrix1, self.matrix2)

        return self.observation, reward, done, {"matrix": self.matrix, "guesses": self.guess_count}

    def _reset(self):
        m1 = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
        m2 = m1.copy()
        self.matrix1 = np.tril(m1) + np.tril(m1, -1).T
        self.matrix2 = np.tril(m2) + np.tril(m2, -1).T
        self.matrix = np.append(self.matrix1.flatten(), self.matrix2.flatten())
        self.guess_count = 0
        self.observation = self.matrix
        return self.observation


class Determinant_v2(gym.Env):
    """Hrabro na problem mini-spektara!
    #TODO: Sta ako je red=kolona? Trebalo bi da je ispravljeno. Skloni 1 sa glavne dijagonale! :/
    """

    def __init__(self):

        self.n = 15

        self.action_space = spaces.Discrete(2 * self.n ** 2)
        self.observation_space = spaces.Box(low=0., high=1., shape=(2 * self.n * self.n))

        m1 = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
        m2 = m1.copy()
        self.matrix1 = np.tril(m1) + np.tril(m1, -1).T
        np.fill_diagonal(self.matrix1, 0)
        self.matrix2 = np.tril(m2) + np.tril(m2, -1).T
        np.fill_diagonal(self.matrix2, 0)
        self.matrix = np.append(self.matrix1.flatten(), self.matrix2.flatten())


        self.g1 = nx.from_numpy_matrix(self.matrix1)
        self.g2 = nx.from_numpy_matrix(self.matrix2)

        self.guess_count = 0
        self.guess_max = 300
        self.observation = self.matrix  # A view of self.matrix, not a new array

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        assert self.action_space.contains(action)

        value_on_action = self.matrix[action]

        reward = 0

        if action < self.n ** 2:
            red = action // self.n
            kolona = action % self.n
            if red!=kolona:
                if value_on_action == 0.:
                    self.matrix1[red, kolona] = 1.
                    self.matrix1[kolona, red] = 1.
                    self.g1.add_edge(red, kolona)
                if value_on_action == 1.:
                    self.matrix1[red, kolona] = 0.
                    self.matrix1[kolona, red] = 0.
                    self.g1.remove_edge(red, kolona)
            else:
                reward = -100

        else:
            action = action - self.n ** 2
            red = action // self.n
            kolona = action % self.n
            if red != kolona:
                if value_on_action == 0.:
                    self.matrix2[red, kolona] = 1.
                    self.matrix2[kolona, red] = 1.
                    self.g2.add_edge(red, kolona)
                if value_on_action == 1.:
                    self.matrix2[red, kolona] = 0.
                    self.matrix2[kolona, red] = 0.
                    self.g2.remove_edge(red, kolona)
            else:
                reward = -100

        self.matrix = np.append(self.matrix1.flatten(), self.matrix2.flatten())



        #d_spektri = razlika_spektara(self.g1, self.g2)
        d_mini_spektri = razlika_mini_spektara(self.g1, self.g2)

        if reward==0:
            reward = 2** (5 - d_mini_spektri)

        #if d_spektri < EPSILON:
        #    reward = -1000

        self.observation = self.matrix

        self.guess_count += 1
        #done = self.guess_count >= self.guess_max or (reward >= 1 - EPSILON and d_spektri >= EPSILON)
        done = self.guess_count >= self.guess_max or (d_mini_spektri<EPSILON)

        if done:
            print(d_mini_spektri)

        return self.observation, reward, done, {"matrix": self.matrix, "guesses": self.guess_count}

    def _reset(self):
        m1 = np.random.randint(0, 2, [self.n, self.n]).astype(np.float32)
        m2 = m1.copy()
        self.matrix1 = np.tril(m1) + np.tril(m1, -1).T
        np.fill_diagonal(self.matrix1, 0)
        self.matrix2 = np.tril(m2) + np.tril(m2, -1).T
        np.fill_diagonal(self.matrix2, 0)
        self.matrix = np.append(self.matrix1.flatten(), self.matrix2.flatten())

        self.g1 = nx.from_numpy_matrix(self.matrix1)
        self.g2 = nx.from_numpy_matrix(self.matrix2)

        self.guess_count = 0
        self.observation = self.matrix
        return self.observation