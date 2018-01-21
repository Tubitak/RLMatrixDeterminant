import numpy as np
import time
from tensorforce.agents.vpg_agent import VPGAgent
from tensorforce.agents.trpo_agent import TRPOAgent
from tensorforce.agents.ppo_agent import PPOAgent
from tensorforce.agents.dqn_agent import DQNAgent
from tensorforce.agents.ddqn_agent import DDQNAgent
from tensorforce.agents.dqn_nstep_agent import DQNNstepAgent
from tensorforce.agents.dqfd_agent import DQFDAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym_det_env



#https://github.com/reinforceio/tensorforce/tree/master/examples/configs

# Create an OpenAIgym environment
env = OpenAIGym('GymDet-v1')

# Network as list of layers
network_spec = [
    {
        "type": "dense",
        "size": 256,
        'activation': 'tanh'
    },

    {
        "type": "dense",
        "size": 128,
        'activation': 'tanh'
    },

    {
        "type": "dense",
        "size": 64,
        'activation': 'tanh'
    }
]

agents = dict(
    vpg_agent=VPGAgent,
    trpo_agent=TRPOAgent,
    ppo_agent=PPOAgent,
    dqn_agent=DQNAgent,
    ddqn_agent=DDQNAgent,
    dqn_nstep_agent=DQNNstepAgent,
    #dqfd_agent=DQFDAgent
)

scores = {}

start_time = time.time()

for agent_class in agents.values():
    print(agent_class)
    env = OpenAIGym('GymDet-v1')

    agent = agent_class(
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network_spec
    )

    runner = Runner(agent=agent, environment=env)


    def episode_finished(r):
        print(
            "Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                   reward=r.episode_rewards[-1]))
        return True


    # Start learning
    runner.run(episodes=1000, max_episode_timesteps=300, episode_finished=None)

    # Print statistics
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=runner.episode,
        ar=np.mean(runner.episode_rewards[-100:]))
    )

    scores[agent_class] = np.mean(runner.episode_rewards[-100:])

print(' ')
print('Time: ', time.time() - start_time)
for agent, score in scores.items():
    print(agent, score)
