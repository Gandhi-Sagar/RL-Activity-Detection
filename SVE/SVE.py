import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import skvideo.io as skio
import cv2

class SingleVideoEnv(gym.Env):
    def __init__(self):
        filename = 'serve.mp4'
        height = 24
        width = 40
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, [height * width])

        self._seed()
        #self._reset()

        self.__videodata = None
        try:
            self.__videodata = skio.vread(filename)
        except FileNotFoundError:
            print(filename + ' could not be opened')
            return False

        with open(filename.split(".")[0] + ".labels", encoding='utf-8') as file:
            l = file.readlines()
        self.__labels = [x.strip() for x in l]

        self.__nb_frames, _, _, _ = self.__videodata.shape
        self.__w = width
        self.__h = height
        self.__index = 0
        self.__correct = 0
        self.__action = 1

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        self.__action = action
        #print('...')
        imgNext, eov_reached = self.__next()

        img = self.__videodata[self.__index - 1, :]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(self.__action), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('window', img)
        cv2.waitKey(1)

        return imgNext, self.__reward(action), eov_reached, {"correct": self.__correct}

    def __next(self):
        self.__index += 1
        if self.__index == self.__nb_frames:
            return np.ones((self.__h*self.__w, ), dtype=np.float), True
        else:
            img = cv2.cvtColor(self.__videodata[self.__index, :], cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.__w, self.__h), interpolation=cv2.INTER_AREA)
            ngray = gray
            ngray = cv2.normalize(gray, ngray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            return ngray.flatten(), False

    def __reward(self, a):
        if a == int(self.__labels[self.__index-1]):
            self.__correct += 1
            if a == 1:
                return 1
            else:
                return 1
        else:
            if a == 1:
                return -1
            else:
                return -1

    def _reset(self):
        self.__index = 0
        self.__correct = 0
        self.__action = 1
        img =  cv2.cvtColor(self.__videodata[self.__index, :], cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.__w, self.__h), interpolation=cv2.INTER_AREA)
        ngray = gray
        ngray = cv2.normalize(gray, ngray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.__index += 1
        return ngray.flatten()

    def _render(self, mode='human', close=False):
        if close:
            return
        img = self.__videodata[self.__index-1, :]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(self.__action), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('window', img)
        cv2.waitKey(1000)
        return
