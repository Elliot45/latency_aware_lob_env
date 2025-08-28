from typing import Tuple
from .utils import Order, Side


class ExecutionModel:
    """
    Modèle d'exécution minimal pour M2 :
    - Fill au mid ± slippage adverse (en bps)
    - Frais proportionnels au notionnel (fee_rate)
    """
    def __init__(self, fee_rate: float = 1e-4, slippage_bps: float = 1.0):
        assert fee_rate >= 0
        assert slippage_bps >= 0
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)

    def _slip_fraction(self) -> float:
        # 10 bps -> 0.001, 1 bps -> 0.0001
        return self.slippage_bps * 1e-4

    def execute(self, order: Order, mid_price: float) -> Tuple[float, float]:
        """
        Args:
            order: Order à exécuter (market par défaut).
            mid_price: mid actuel.
        Returns:
            fill_price, fees
        """
        if not order.is_trade():
            return mid_price, 0.0

        slip = self._slip_fraction()
        if order.side == Side.BUY:
            fill = mid_price * (1.0 + slip)
        else:  # SELL
            fill = mid_price * (1.0 - slip)

        fees = self.fee_rate * abs(order.size) * fill
        return float(fill), float(fees)
