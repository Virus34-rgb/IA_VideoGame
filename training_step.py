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
    stats_path: Optional[str] = None
    rusher_aggression: Optional[float] = None
    rusher_aggression_min: Optional[float] = None 
    rusher_aggression_max: Optional[float] = None 