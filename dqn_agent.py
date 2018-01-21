import numpy as np
import time
import json
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym_det_env


# Create an OpenAIgym environment
env = OpenAIGym('GymDet-v2')

# Network as list of layers
network_spec = [
    {
        "type": "dense",
        "size": 1024
    },

    {
        "type": "dense",
        "size": 512
    },

    {
        "type": "dense",
        "size": 256
    }
]

# Probaj i koji konvolucioni sloj!


with open('agents/dqn.json', 'r') as fp:
    agent_config = json.load(fp=fp)

agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states_spec=env.states,
            actions_spec=env.actions,
            network_spec=network_spec
        )
    )

start_time = time.time()


runner = Runner(agent=agent, environment=env)


def episode_finished(r):
    print(
        "Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                               reward=r.episode_rewards[-1]))
    return True

# Start learning

runner.run(episodes=2000, max_episode_timesteps=300, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 1000 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-1000:]))
)

agent.save_model('agents/dqn_agent_trained.MODEL', append_timestep=True)

print(' ')
print('Time: ', time.time() - start_time)
