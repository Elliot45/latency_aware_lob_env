from collections import deque
from typing import Deque, List, Tuple, Optional
from .utils import Order


class LatencyQueue:
    """
    File d'attente simple : tout ordre ajouté est exécuté après `delay` ticks.
    Possibilité de spécifier un `delay` par ordre.
    """
    def __init__(self, default_delay: int = 5):
        assert default_delay >= 0
        self.default_delay = default_delay
        self._q: Deque[Tuple[Order, int]] = deque()

    def __len__(self) -> int:
        return len(self._q)

    def reset(self) -> None:
        self._q.clear()

    def add(self, order: Order, delay: Optional[int] = None) -> None:
        d = self.default_delay if delay is None else max(0, int(delay))
        self._q.append((order, d))

    def process(self) -> List[Order]:
        """
        Décrémente les délais, renvoie la liste des ordres dont le délai atteint 0.
        """
        executed: List[Order] = []
        for _ in range(len(self._q)):
            order, t = self._q.popleft()
            t -= 1
            if t <= 0:
                executed.append(order)
            else:
                self._q.append((order, t))
        return executed
