
import os
from pathlib import Path
import random
import re

import torch

from constants import MAX_MODELS


class OpponentPool:
    def __init__ (self,path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        
    def save_version(self,player):
        cantidad,first_index,last_index = self.list_models()
        if(cantidad >= MAX_MODELS):
            self.delete_first(first_index)
        path1 = self.path / f"snapshotsSELECTION_{last_index + 1}.pth"
        path2 = self.path / f"snapshotsTURN_{last_index + 1}.pth"
        for path, (net, target_net, opt, eps_attr, replayed_attr) in zip((path1, path2), player._network_specs()):
            torch.save({
                "dqn": net.state_dict(),
                "targetdqn": target_net.state_dict(),
                "optimizer": opt.state_dict(),
                "epsilon": getattr(player, eps_attr),
                "replayed_selection": getattr(player, replayed_attr),
            }, path)
            
    def get_random(self):
        _,first_index,last_index = self.list_models()
        index = random.randint(first_index,last_index)
        path1 = self.path / f"snapshotsSELECTION_{index}.pth"
        path2 = self.path / f"snapshotsTURN_{index}.pth"
        return path1,path2
    
    def list_models(self):
        indexes = []

        for path in self.path.glob("snapshotsSELECTION_*.pth"):
            match = re.search(r'_(\d+)\.pth$', path.name)
            if match:
                indexes.append(int(match.group(1)))

        if not indexes:
            return 0, 0, 0

        return len(indexes), min(indexes), max(indexes)
    
    def delete_first(self,first):
        path1 = self.path / f"snapshotsSELECTION_{first}.pth"
        path2 = self.path / f"snapshotsTURN_{first}.pth"
        archive1 = Path(path1)
        archive2 = Path(path2)
        archive1.unlink()
        archive2.unlink()