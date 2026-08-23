# config.py
from dataclasses import dataclass, field
import os


@dataclass
class RunConfig:
    version: int
    train_episodes: int = 50000
    eval_episodes: int = 5000
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    @property
    def base_path(self):
        return os.path.join(self.base_dir, "models", f"IAV{self.version}")

    @property
    def p1_path(self):
        return os.path.join(self.base_path, "P1")

    @property
    def p2_path(self):
        return os.path.join(self.base_path, "P2")

    @property
    def stats_path(self):
        return os.path.join(self.base_path, "stats.txt")

    @property
    def stats2_path(self):
        return os.path.join(self.base_path, "stats2.txt")

    @property
    def path_p1_sel(self):
        return os.path.join(self.p1_path, "Disp.pth")

    @property
    def path_p1_turn(self):
        return os.path.join(self.p1_path, "Act.pth")

    @property
    def path_p2_sel(self):
        return os.path.join(self.p2_path, "Disp.pth")

    @property
    def path_p2_turn(self):
        return os.path.join(self.p2_path, "Act.pth")
    
    @property
    def path_opp_pool(self):
        return os.path.join(self.base_path,"opponent_pool")