import numpy as np
from lob_env.env import LatencyAwareLOBEnv


class FakeLOB:
    """
    LOB factice pour tests: mid fixe, état zéro, jamais 'done'.
    """
    def __init__(self, depth=10):
        self.depth = depth
        self.done = False
        self._mid = 100.0

    def reset(self):
        self.done = False
        self._mid = 100.0

    def step(self):
        # on peut injecter une mini dynamique si besoin
        pass

    def get_state(self):
        # vecteur de taille 2*depth
        return np.zeros(2 * self.depth, dtype=np.float32)

    def get_mid_price(self):
        return self._mid


def test_env_shapes_and_step():
    env = LatencyAwareLOBEnv(depth=10, external_lob=FakeLOB(depth=10),
                             latency_ticks=2, slippage_bps=1.0, fee_rate=1e-4)

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    expected_dim = env.observation_space.shape[0]
    assert obs.shape == (expected_dim,)
    assert env.action_space.n == 3

    # HOLD -> pas d'exécution immédiate
    step_out = env.step(0)
    if len(step_out) == 5:  # gymnasium
        obs2, reward, terminated, truncated, info = step_out
    else:
        obs2, reward, terminated, info = step_out

    assert obs2.shape == (expected_dim,)
    assert isinstance(reward, float)
    assert not terminated
    assert "queue_len" in info

    # BUY -> ordre part dans la file, ne s'exécutera qu'après 2 ticks
    env.step(1)
    assert env.queue.default_delay == 2

