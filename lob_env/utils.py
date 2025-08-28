from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Side(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = -1


@dataclass(frozen=True)
class Order:
    side: Side
    size: float = 1.0
    kind: str = "market"  # "market" | "limit" (placeholder pour M3+)
    limit_price: Optional[float] = None  # utilisé si kind == "limit"

    def is_trade(self) -> bool:
        return self.side in (Side.BUY, Side.SELL)
