# Disclaimer

Module `agent` is a bit of a mess at the moment because it mixes two different coding approaches, that I haven't yet made my mind up to reconcile:

1. DDQN, VPG, A2C (with GAE): one algorithm = one file = one class, with methods that define everything that's needed to train the model.
2. PPO: one algorithm = files for agent, buffer, and run, with functions separating properties of the agent vs of the run (e.g. training schedule).
