# Lab 2: Playing OpenAI GYM games
# OpenAI GYM
import msvcrt
import gym
from gym.envs.registration import register

class _Getch:
    def __call__(self):
        return msvcrt.getch()

inkey = _Getch()

# Macros
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    b'w': UP,
    b's': DOWN,
    b'd': RIGHT,
    b'a': LEFT}

def main():
    # Register FrozenLake
    register(
        id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False})

    env = gym.make('FrozenLake-v3')
    env.render() # Show the initial board

    while True:
        # Choose an action from keyboard
        key = inkey()
        print(key)
        if key not in arrow_keys.keys():
            print("Game aborted!")
            break

        action = arrow_keys[key]
        state, reward, done, info = env.step(action)
        env.render() # Show the board after action
        print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

        if done:
            print("finished with reward", reward)
            break

if __name__ == '__main__':
    main()