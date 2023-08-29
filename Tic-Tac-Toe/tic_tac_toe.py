import numpy as np
import gym
from gym import spaces
from PIL import Image
import matplotlib.pyplot as plt

STATE_W = 150
STATE_H = 150


class TicTacToe(gym.Env):
    def __init__(self, player):
        self.board = np.zeros((3, 3))
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_W, STATE_H, 3), dtype=np.uint8)
        self.player = player

    def step(self, action):
        if self.board[action // 3][action % 3] != 0:
            return self.draw_figures(), -1, True, {}

        self.board[action // 3][action % 3] = self.player

        if self.check_winner(self.player):
            return self.draw_figures(), 1, True, {}

        if self.check_winner(-self.player):
            return self.draw_figures(), -1, True, {}

        if self.is_board_full():
            return self.draw_figures(), 0, True, {}

        opponent_action = np.random.randint(0, 9)
        while self.board[opponent_action // 3][opponent_action % 3] != 0:
            opponent_action = np.random.randint(0, 9)

        self.board[opponent_action // 3][opponent_action % 3] = -self.player

        return self.draw_figures(), 0, False, {}

    def reset(self):
        self.board = np.zeros((3, 3))
        if self.player == -1:
            opponent_action = np.random.randint(0, 9)
            self.board[opponent_action // 3][opponent_action %
                                             3] = -self.player

        return self.draw_figures()

    def draw_figures(self):
        x = Image.open('data/x.png').convert('RGB')
        o = Image.open('data/o.png').convert('RGB')

        if x.size != o.size:
            raise ValueError("two images should have the same size")

        size = x.size
        output = np.zeros((3*size[0], 3*size[1], 3))

        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 1:
                    output[i*size[0]:i*size[0]+size[0],
                           j*size[1]:j*size[1]+size[1]] = x
                if self.board[i, j] == -1:
                    output[i*size[0]:i*size[0]+size[0],
                           j*size[1]:j*size[1]+size[1]] = o

        output[size[0]-5:size[0]+5, :] = 255
        output[2*size[0]-5:2*size[1]+5, :] = 255
        output[:, size[1]-5:size[1]+5] = 255
        output[:, 2*size[1]-5:2*size[1]+5] = 255

        output = Image.fromarray(output.astype(np.uint8))
        output = output.resize((STATE_W, STATE_H))
        output = np.array(output)

        return output

    def render(self, mode="human"):
        plt.imshow(self.draw_figures()/255)
        plt.show()

    def check_winner(self, player):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] == player:
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] == player:
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == player:
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] == player:
            return True
        return False

    def is_board_full(self):
        return np.all(self.board != 0)
    

# env = TicTacToe(1)
# observation = env.reset()
# done = False

# while not done:
#     env.render()
#     action = int(input("Enter your action (0-8): "))
#     observation, reward, done, _ = env.step(action)
#     print("Reward:", reward)

# env.render()