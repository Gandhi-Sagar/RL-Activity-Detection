import skvideo.io as skio
import cv2


class SingleVideoEnv:
    def __init__(self, filename):
        __videodata = None
        try:
            __videodata = skio.vread(filename)
        except FileNotFoundError:
            print(filename + ' could not be opened')
            return False
        __nb_frames, __h, __w, _ = __videodata.shape

    def reset(self):
        # grab the next frame
        # then retrieve and return
        pass

    def next(self):
        # see if there exists a next frame
        # if yes, return it, o.w. return false
        # this method can be private
        pass
    def reward(self):
        # this method is private
        # it covers all the possibilities of action
        # must know labels and then it is possible to assign the reward
        pass
    def step(self, taken_action):
        # return next frame,
        # return reward associated with prev frame for "taken_action"
        # return true if there is no next frame
        pass
    class action_space:
        def __init__(self):
            pass
        def contains(self):
            # return true for number within 0 to num_possible_states - 1
            # o.w. false
            pass
        def sample(self):
            #return a number at random in range 0 to num_possible_states - 1
            pass