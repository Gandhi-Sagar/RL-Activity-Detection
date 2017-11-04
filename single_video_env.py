import skvideo.io as skio
import cv2
import random


class SingleVideoEnv:
    def __init__(self, filename, width, height):
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

    def reset(self):
        # return the 0'th frame, advance the counter
        self.__index = 0
        img =  cv2.cvtColor(self.__videodata[self.__index, :], cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.__w, self.__h), interpolation=cv2.INTER_AREA)
        ngray = gray
        cv2.normalize(gray, ngray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.__index += 1
        return ngray

    def step(self, taken_action):
        # return next frame,
        # return reward associated with prev frame for "taken_action"
        # return true if there is no next frame
        imgNext = self.__next()
        eov_reached = False
        if imgNext is None:
            eov_reached = True
        return imgNext, self.__reward(taken_action), eov_reached, self.__correct

    def __next(self):
        # see if there exists a next frame
        # if yes, return it, o.w. return false
        # this method can be private
        self.__index += 1
        if self.__index == self.__nb_frames:
            return None
        else:
            img = cv2.cvtColor(self.__videodata[self.__index, :], cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.__w, self.__h), interpolation=cv2.INTER_AREA)
            ngray = gray
            cv2.normalize(gray, ngray, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            return ngray

    def __reward(self, a):
        # this method is only suitable for this env
        # it covers all the possibilities of action
        # must know labels and then it is possible to assign the reward
        if a == self.__labels[self.__index-1]:
            self.__correct += 1
            if a == 1:
                return 10
            else:
                return 100
        else:
            if a == 1:
                return -50
            else:
                return -100

    # as of now, action space only has two actions, content = 1, non-content = 0
class action_space:
    def __init__(self):
        self.__possible_actions = 2

    def contains(self, action):
        # return true for number within 0 to num_possible_actions - 1
        # o.w. false
        if action >= 0 and action < self.__possible_actions:
            return True
        else:
            return False

    def sample(self):
        #return a number at random in range 0 to num_possible_states - 1
        return random.randint(0, (self.__possible_actions - 1))