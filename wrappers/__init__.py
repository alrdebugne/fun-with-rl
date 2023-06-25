"""gym.Wrappers to accelerate learning"""
from .wrappers import (
    make_env,
    make_nes_env,
    MaxAndSkipEnv,
    ProcessFrame84,
    ProcessFrame84Segment,
    ImageToPyTorch,
    BufferWrapper,
    ScaledFloatFrame,
)
