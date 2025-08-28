from typing import Optional, Dict, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Support gymnasium d'abord, fallback sur gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except Exception:  # pragma: no cover
    import gym
    from gym import spaces
    _GYMNASIUM = False

from .latency_queue import LatencyQueue
from .execution_costs import ExecutionModel
from .utils import Order, Side


class LatencyAwareLOBEnv(gym.Env):
    """
    Environnement RL basé LOB + latence.
    Hypothèses M2 :
      - observation = état LOB (vectorisé) fourni par le simulateur (Projet 1)
      - action_space Discret: 0=HOLD, 1=BUY (market), 2=SELL (market)
      - reward = Δ(equity) = (cash + inv*mid)_t - (...)_{t-1}
      - exécution: mid ± slippage bps + frais (fee_rate)
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        depth: int = 10,
        latency_ticks: int = 5,
        latency_jitter: int = 0,
        fee_rate: float = 1e-4,
        slippage_bps: float = 1.0,
        max_episode_steps: int = 20_000,
        trade_size: float = 1.0,
        alpha_strength: float = 0.02,
        inactivity_penalty: float = 0.0,
        external_lob: Optional[object] = None,
        seed: Optional[int] = None,
        include_features: Tuple[str, ...] = ("lob", "mid", "inventory", "cash", "queue_len"),
        normalize: bool = True,
        cash_scale: float = 1e5,
        inventory_scale: float = 100.0,
        mid_scale: float = 1e3,
        queue_scale: float = 100.0,
        lob_scale: float = 1e3,
    ):
        super().__init__()

        # LOB du Projet 1
        if external_lob is not None:
            self.lob = external_lob
        else:
            # Import paresseux pour ne pas forcer la dépendance dans les tests
            from lob_simulator_core import LOBSimulator, LOBConfig  # type: ignore
            self.lob = LOBSimulator(config=LOBConfig(depth_levels=depth))

        self.depth = int(depth)
        self.queue = LatencyQueue(default_delay=latency_ticks)
        self.latency_ticks = int(latency_ticks)
        self.latency_jitter = int(latency_jitter)
        self.exec_model = ExecutionModel(fee_rate=fee_rate, slippage_bps=slippage_bps)
        self.max_episode_steps = int(max_episode_steps)
        self.trade_size = float(trade_size)
        self.alpha_strength = float(alpha_strength)
        self.inactivity_penalty = float(inactivity_penalty)

        # --- définir ces attributs AVANT de construire observation_space ---
        self.include_features = tuple(include_features)
        self.normalize = bool(normalize)
        self.cash_scale = float(cash_scale)
        self.inventory_scale = float(inventory_scale)
        self.mid_scale = float(mid_scale)
        self.queue_scale = float(queue_scale)
        self.lob_scale = float(lob_scale)


        # RNG
        self._np_rng = np.random.default_rng(seed)
        self._steps = 0

        # Spaces dynamiques selon les features choisies
        lob_dim = 2 * self.depth if "lob" in self.include_features else 0
        extra_dim = 0
        extra_dim += 1 if "mid" in self.include_features else 0
        extra_dim += 1 if "inventory" in self.include_features else 0
        extra_dim += 1 if "cash" in self.include_features else 0
        extra_dim += 1 if "queue_len" in self.include_features else 0

        obs_dim = lob_dim + extra_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

        # Comptes
        self.cash = 0.0
        self.inventory = 0.0
        self._prev_equity = 0.0


    # --- Gym API ---
    def seed(self, seed: Optional[int] = None) -> None:  # gym classic
        self._np_rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)

        # Réinitialise LOB, queue, comptes
        if hasattr(self.lob, "reset"):
            self.lob.reset()
        self.queue.reset()
        self.cash = 0.0
        self.inventory = 0.0
        self._steps = 0

        obs = self._get_observation()
        mid = self._get_mid()
        self._prev_equity = self._equity(mid)

        if _GYMNASIUM:
            return obs.astype(np.float32), {}
        return obs.astype(np.float32)

    def step(self, action: int):
        self._steps += 1

        # 1) Action -> order (eventuel)
        order = self._action_to_order(action)

        # 2) Place ordre dans la file (latence + jitter)
        if order is not None:
            delay = int(self.latency_ticks)
            if self.latency_jitter > 0:
                jitter = int(self._np_rng.integers(-self.latency_jitter, self.latency_jitter + 1))
                delay = max(0, delay + jitter)
            self.queue.add(order, delay=delay)

        # 3) Exécuter ordres arrivés à échéance
        executed = self.queue.process()
        mid_before = self._get_mid()
        for ex in executed:
            fill, fees = self.exec_model.execute(ex, mid_before)
            notional = abs(ex.size) * fill
            if ex.side == Side.BUY:
                self.cash -= notional + fees
                self.inventory += ex.size
            elif ex.side == Side.SELL:
                self.cash += notional - fees
                self.inventory -= ex.size

        # 4) Avancer LOB d'un tick
        if hasattr(self.lob, "step"):
            self.lob.step()

        # 5) Observation + reward
        obs = self._get_observation()
        mid = self._get_mid()
        equity = self._equity(mid)
        reward = equity - self._prev_equity  # PnL net des coûts & slippage
        self._prev_equity = equity

        # 5.b) Micro-alpha: Order Book Imbalance (OBI) → petite prime directionnelle
        # OBI ∈ [-1, 1] ; si tu as des helpers best(), utilise-les, sinon on lit la structure.
        bid_vol_L1 = 0.0
        ask_vol_L1 = 0.0
        try:
            # si OrderBookSide expose best() -> (price, volume)
            _, bid_vol_L1 = self.lob.bids.best()
            _, ask_vol_L1 = self.lob.asks.best()
        except Exception:
            # fallback sur tes structures internes
            if getattr(self.lob.bids, "_prices_sorted", None):
                p_bid = max(self.lob.bids._prices_sorted)
                bid_vol_L1 = float(self.lob.bids.levels.get(p_bid, 0.0))
            if getattr(self.lob.asks, "_prices_sorted", None):
                p_ask = min(self.lob.asks._prices_sorted)
                ask_vol_L1 = float(self.lob.asks.levels.get(p_ask, 0.0))

        den = bid_vol_L1 + ask_vol_L1
        if den <= 1e-6:
            obi = 0.0
        else:
            obi = (bid_vol_L1 - ask_vol_L1) / den
            # clamp par prudence numérique
            if obi > 1.0: obi = 1.0
            if obi < -1.0: obi = -1.0

        edge = self.alpha_strength * obi
        if action == 1:        # BUY (market)
            reward += edge
        elif action == 2:      # SELL (market)
            reward -= edge

        # 5.c) (optionnel) légère pénalité d'inactivité pour éviter HOLD absolu
        if self.inactivity_penalty > 0.0 and action == 0:  # HOLD
            reward -= self.inactivity_penalty

        # garde-fous numériques
        if not np.isfinite(reward):
         reward = 0.0
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # 6) Terminaison
        done_sim = bool(getattr(self.lob, "done", False))
        done_len = self._steps >= self.max_episode_steps
        terminated = done_sim or done_len

        info = {
            "mid": float(mid),
            "cash": float(self.cash),
            "inventory": float(self.inventory),
            "equity": float(equity),
            "queue_len": len(self.queue),
            "executed_count": len(executed),
            "obi": float(obi),
        }

        if _GYMNASIUM:
            # Gymnasium: (obs, reward, terminated, truncated, info)
            truncated = False  # on peut affiner si besoin
            return obs.astype(np.float32), float(reward), terminated, truncated, info
        else:
            # Gym classique: (obs, reward, done, info)
            return obs.astype(np.float32), float(reward), terminated, info

    # --- Helpers ---
    def _get_observation(self) -> np.ndarray:
        parts = []

        # LOB vectorisé
        if "lob" in self.include_features:
            if hasattr(self.lob, "get_state"):
                lob_vec = np.asarray(self.lob.get_state(), dtype=np.float32).reshape(-1)
                target = 2 * self.depth
                if lob_vec.size < target:
                    pad = np.zeros(target - lob_vec.size, dtype=np.float32)
                    lob_vec = np.concatenate([lob_vec, pad], axis=0)
                elif lob_vec.size > target:
                    lob_vec = lob_vec[:target]
            else:
                lob_vec = np.zeros((2 * self.depth,), dtype=np.float32)
            if self.normalize and self.lob_scale > 0:
                lob_vec = lob_vec / self.lob_scale
            parts.append(lob_vec.astype(np.float32))

        # Mid
        if "mid" in self.include_features:
            mid = self._get_mid()
            val = mid / self.mid_scale if (self.normalize and self.mid_scale > 0) else mid
            parts.append(np.array([val], dtype=np.float32))

        # Inventory
        if "inventory" in self.include_features:
            inv = self.inventory
            val = inv / self.inventory_scale if (self.normalize and self.inventory_scale > 0) else inv
            parts.append(np.array([val], dtype=np.float32))

        # Cash
        if "cash" in self.include_features:
            c = self.cash
            val = c / self.cash_scale if (self.normalize and self.cash_scale > 0) else c
            parts.append(np.array([val], dtype=np.float32))

        # Queue length
        if "queue_len" in self.include_features:
            q = float(len(self.queue))
            val = q / self.queue_scale if (self.normalize and self.queue_scale > 0) else q
            parts.append(np.array([val], dtype=np.float32))

        if parts:
            return np.concatenate(parts, axis=0).astype(np.float32)
        # fallback
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_mid(self) -> float:
        if hasattr(self.lob, "get_mid_price"):
            return float(self.lob.get_mid_price())
        # fallback: mid fictif stable
        return 100.0

    def _equity(self, mid: float) -> float:
        return self.cash + self.inventory * mid

    def _action_to_order(self, action: int) -> Optional[Order]:
        if int(action) == 1:
            return Order(side=Side.BUY, size=self.trade_size, kind="market")
        elif int(action) == 2:
            return Order(side=Side.SELL, size=self.trade_size, kind="market")
        return None  # HOLD

    
    # --- Optionnel: rendu ---
    def render(self, mode: str = "human") -> None:
        """
        Rendu simple :
        - mode="human": print infos clés
        - mode="ansi": renvoie une string
        """
        mid = self._get_mid()
        equity = self._equity(mid)
        msg = (
            f"Step={self._steps} | "
            f"Mid={mid:.2f} | Cash={self.cash:.2f} | "
            f"Inv={self.inventory:.2f} | Equity={equity:.2f} | "
            f"Queue={len(self.queue)}"
        )
        if mode == "ansi":
            return msg
        else:
            print(msg)

    def close(self) -> None:
        """
        Ici, rien de spécial (pas de thread, pas de fenêtre).
        Placeholder pour compatibilité gym/gymnasium.
        """
        pass

