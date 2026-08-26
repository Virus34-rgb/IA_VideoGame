import random
from pathlib import Path
import re
import torch

from constants import MAX_MODELS


class OpponentPoolV:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._player_cache = {}  # {checkpoint_id: PlayerAIV cargado}
        self._indices = self._scan_disk_once()

    def _scan_disk_once(self):
        indexes = []
        for path in self.path.glob("snapshotsSELECTION_*.pth"):
            match = re.search(r'_(\d+)\.pth$', path.name)
            if match:
                indexes.append(int(match.group(1)))
        return sorted(indexes)

    def save_version(self, player):
        cantidad, first_index, last_index = self.list_models()
        if cantidad >= MAX_MODELS:
            self.delete_first(first_index)
            cantidad, first_index, last_index = self.list_models()
        new_index = last_index + 1 if cantidad > 0 else 1
        path1 = self.path / f"snapshotsSELECTION_{new_index}.pth"
        path2 = self.path / f"snapshotsTURN_{new_index}.pth"
        player.save_model_inference_only(path1, path2)
        self._indices.append(new_index)

    def get_random(self):
        _, first_index, last_index = self.list_models()
        index = random.randint(first_index, last_index)
        return self.path / f"snapshotsSELECTION_{index}.pth", self.path / f"snapshotsTURN_{index}.pth"

    def list_models(self):
        if not self._indices:
            return 0, 0, 0
        return len(self._indices), self._indices[0], self._indices[-1]

    def delete_first(self, first):
        (self.path / f"snapshotsSELECTION_{first}.pth").unlink()
        (self.path / f"snapshotsTURN_{first}.pth").unlink()
        self._player_cache.pop(first, None)
        self._indices.remove(first)

    def sample_assignment(self, N, pool_porcentage):
        cantidad, first_index, last_index = self.list_models()
        from_pool = torch.rand(N) < pool_porcentage
        if cantidad == 0:
            from_pool[:] = False

        checkpoint_idx = torch.full((N,), -1, dtype=torch.long)
        if cantidad > 0:
            n_from_pool = from_pool.sum().item()
            if n_from_pool > 0:
                elegidos = torch.randint(first_index, last_index + 1, (n_from_pool,))
                checkpoint_idx[from_pool] = elegidos
        return from_pool, checkpoint_idx

    def build_grouped_opponents(self, checkpoint_idx, player_class, N, environment):
        grupos = {}
        unicos = checkpoint_idx.unique()
        for cp_id in unicos.tolist():
            if cp_id == -1:
                continue
            idx_partidas = (checkpoint_idx == cp_id).nonzero(as_tuple=True)[0]
            if cp_id not in self._player_cache:
                jugador = player_class(N, environment)
                path1 = self.path / f"snapshotsSELECTION_{cp_id}.pth"
                path2 = self.path / f"snapshotsTURN_{cp_id}.pth"
                jugador.load_model_inference_only(path1, path2)
                self._player_cache[cp_id] = jugador
            grupos[cp_id] = (self._player_cache[cp_id], idx_partidas)
        return grupos