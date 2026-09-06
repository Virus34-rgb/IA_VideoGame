from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class TrainingStep:
    name: str
    action: str
    episodes: int
    opponent_factory: Callable
    load_opponent_checkpoint: Optional[tuple] = None
    player1_checkpoint: Optional[tuple] = None  
    epsilon_turn: Optional[float] = None         
    epsilon_sel: Optional[float] = None           
    learn_p1: Optional[bool] = None           
    learn_p2: Optional[bool] = None           
    stats_path: Optional[str] = None   # ruta de stats a usar en este step; si es None, se usa el fallback de siempre (config.stats_path / config.stats2_path segun action)