import pytest
from lob_env.latency_queue import LatencyQueue
from lob_env.utils import Order, Side


def test_latency_queue_basic():
    q = LatencyQueue(default_delay=2)
    q.add(Order(side=Side.BUY, size=1.0))
    assert len(q) == 1

    # t=1 -> rien n'est exécuté
    executed = q.process()
    assert executed == []
    assert len(q) == 1

    # t=2 -> l'ordre est exécuté
    executed = q.process()
    assert len(executed) == 1
    assert len(q) == 0

    # delay par ordre
    q.add(Order(side=Side.SELL, size=1.0), delay=0)
    executed = q.process()
    assert len(executed) == 1
