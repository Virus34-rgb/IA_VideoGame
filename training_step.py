from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class TrainingStep:
    name: str
    action: str
    episodes: int
    opponent_factory: Callable
    load_opponent_checkpoint: Optional[tuple] = None
    player1_checkpoint: Optional[tuple] = None   # nuevo: Opción B
    epsilon_turn: Optional[float] = None          # nuevo
    epsilon_sel: Optional[float] = None            # nuevo
    learn_p1: Optional[bool] = None                # nuevo
    learn_p2: Optional[bool] = None                # nuevo