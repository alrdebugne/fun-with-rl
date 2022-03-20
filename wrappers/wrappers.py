"""
Wrappers used in tutorial
Based on OpenAI baselines: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
import collections
import numpy as np
import gym
import cv2


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """
        Return only every `skip`-th frame, max-pooling over last two frames.
        Max-pooling is performed to avoid learning on flickers that affect
        some older games.
        """
        super(MaxAndSkipEnv, self).__init__(env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        # ^ think of deque-s like lists with O(1) for pop and append
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # Max-pool across last two time steps (i.e. keep max value at every pixel x channel)
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsample image to 84 x 84 & greyscale image
    Currently only works if the input image is of size (240, 256, 3)
    """

    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        # ^ factors used across Atari games for greyscaling (not sure why)
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image format from (height, width, channel) to PyTorch's
    expected format of (channel, height, width)
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalise pixel values in frame from [0, 255] to [0, 1]"""

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    """
    In many video games, observing a single frame does not convey sufficient game
    information (e.g. not knowing where the ball is moving in Pong). To circumvent
    this problem, we define a state as consisting of `n_steps` stacked images.
    """

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype,
        )

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """Update self.buffer by shifting old frames left, then adding new frame"""
        self.buffer[:-1] = self.buffer[1:]  # shift old frames left
        self.buffer[-1] = observation  # add new frame
        return self.buffer
