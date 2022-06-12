"""gym.Wrappers to accelerate learning"""
from .wrappers import (
    make_env,
    MaxAndSkipEnv,
    ProcessFrame84,
    ImageToPyTorch,
    BufferWrapper,
    ScaledFloatFrame,
)
