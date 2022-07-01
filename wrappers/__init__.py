"""gym.Wrappers to accelerate learning"""
from .wrappers import (
    make_env,
    make_nes_env,
    MaxAndSkipEnv,
    ProcessFrame84,
    ImageToPyTorch,
    BufferWrapper,
    ScaledFloatFrame,
)
