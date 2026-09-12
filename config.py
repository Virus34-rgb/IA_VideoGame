from dataclasses import dataclass, field
import os
import re
from typing import Optional


@dataclass
class RunConfig:
    version: int
    train_episodes: int = 50000
    eval_episodes: int = 5000
    suffix: str = ""   # <--- NUEVO
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    base_path_override: Optional[str] = None   # ← NUEVO

    @property
    def base_path(self):
        if self.base_path_override:            # ← NUEVO
            return self.base_path_override
        if self.suffix:
            clean_suffix = self.sanitize_filename(self.suffix)
            folder = f"IAV{self.version}_{clean_suffix}"
        else:
            folder = f"IAV{self.version}"
        return os.path.join(self.base_dir, "models", folder)

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
    def stats_rusher_path(self):
        return os.path.join(self.base_path, "stats_rusher.txt")
    
    @property
    def stats_rusher_aggr_low_path(self):
        return os.path.join(self.base_path, "stats_rusher_aggr_0.txt")

    @property
    def stats_rusher_aggr_mid_path(self):
        return os.path.join(self.base_path, "stats_rusher_aggr_05.txt")

    @property
    def stats_rusher_aggr_high_path(self):
        return os.path.join(self.base_path, "stats_rusher_aggr_1.txt")
    
    @property
    def stats_rusher_finetune_low(self):
        return os.path.join(self.base_path, "stats_rusher_finetune_low.txt")
    @property
    def stats_rusher_finetune_medium(self):
        return os.path.join(self.base_path, "stats_rusher_finetune_medium.txt")
    @property
    def stats_rusher_finetune_hight(self):
        return os.path.join(self.base_path, "stats_rusher_finetune_hight.txt")
    
    @property
    def stats_human(self):
        return os.path.join(self.base_path, "stats_human.txt")
    
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
        return os.path.join(self.base_path, "opponent_pool")
    
    def sanitize_filename(self,name: str) -> str:
        """Reemplaza caracteres no válidos en nombres de archivo por '_'."""
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)