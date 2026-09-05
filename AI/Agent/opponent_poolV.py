"""
Pool de oponentes entrenados (snapshots de P2).

Almacena modelos en disco, los carga bajo demanda y permite muestrear
para enfrentarlos contra el agente principal (P1).
Preparado para futura extensión con matchmaking por Elo.
"""
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch

import constants


class OpponentPoolV:
    """
    Gestiona una colección de modelos de oponentes (snapshots de P2).

    Los modelos se guardan como archivos .pth en el directorio especificado.
    Permite guardar nuevas versiones, listar existentes y muestrear
    aleatoriamente para asignar oponentes a partidas paralelas.
    """

    def __init__(self, path: str) -> None:
        """
        Args:
            path: Ruta al directorio donde se almacenan los snapshots.
        """
        self.path: Path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        # Cache de instancias de PlayerAIV cargadas para reutilización entre partidas
        self._player_cache: Dict[int, Any] = {}
        self._indices: List[int] = self._scan_disk_once()
        self.elos: Dict[int, float] = self._load_elos()
        self.elos = {k: v for k, v in self.elos.items() if k in self._indices}

    def _scan_disk_once(self) -> List[int]:
        """
        Escanea el directorio en busca de snapshots existentes y guarda sus índices.

        Returns:
            Lista ordenada de índices disponibles.
        """
        indexes = []
        for file_path in self.path.glob("snapshotsSELECTION_*.pth"):
            match = re.search(r"_(\d+)\.pth$", file_path.name)
            if match:
                indexes.append(int(match.group(1)))
        return sorted(indexes)
    
    def _elo_file(self) -> Path:
        """Ruta del archivo donde se persisten los ratings Elo del pool."""
        return self.path / "elo_ratings.json"

    def _load_elos(self) -> Dict[int, float]:
        """Carga los ratings Elo desde disco, si el archivo existe."""
        elo_path = self._elo_file()
        if not elo_path.exists():
            return {}
        with open(elo_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Las claves JSON son siempre str; hay que convertirlas a int
        return {int(k): float(v) for k, v in raw.items()}

    def _save_elos(self) -> None:
        """Persiste el diccionario de ratings Elo a disco."""
        with open(self._elo_file(), "w", encoding="utf-8") as f:
            json.dump(self.elos, f, indent=2)
        
    def save_version(self, player: Any) -> None:
        """
        Guarda el estado actual del jugador como un nuevo snapshot en la pool.

        Si se alcanza el límite MAX_MODELS, elimina el más antiguo primero.

        Args:
            player: Instancia de PlayerAIV (o cualquier objeto con save_model_inference_only).
        """
        # CAMBIADO: guard defensivo. save_version() solo tiene sentido para
        # jugadores con red neuronal entrenable (PlayerAIV); PlayerGUIV y
        # PlayerNoAIV no implementan save_model_inference_only porque no hay
        # nada que snapshotear en un humano. Antes esto asumía ciegamente que
        # player siempre era un PlayerAIV, y crasheaba con AttributeError si
        # se llamaba (por ejemplo, si el guard de learn_p1/learn_p2 en
        # TrainerV._run no cubría el caso exacto). Ahora, si el jugador no
        # soporta snapshots, la función simplemente no hace nada.
        if not hasattr(player, "save_model_inference_only"):
            return

        cantidad, first_index, last_index = self.list_models()
        if cantidad >= constants.MAX_MODELS:
            self.delete_first(first_index)
            cantidad, first_index, last_index = self.list_models()

        new_index = last_index + 1 if cantidad > 0 else 1
        path_sel = self.path / f"snapshotsSELECTION_{new_index}.pth"
        path_turn = self.path / f"snapshotsTURN_{new_index}.pth"

        player.save_model_inference_only(path_sel, path_turn)
        self._indices.append(new_index)
        self.elos[new_index] = player.elo
        self._save_elos()


    def get_random(self) -> Tuple[Path, Path]:
        """
        Devuelve las rutas de un snapshot aleatorio.

        Returns:
            Tupla (path_selection, path_turn) para cargar el modelo.
        """
        _, first_index, last_index = self.list_models()
        index = random.randint(first_index, last_index)
        return self.path / f"snapshotsSELECTION_{index}.pth", self.path / f"snapshotsTURN_{index}.pth"

    def list_models(self) -> Tuple[int, int, int]:
        """
        Devuelve información sobre los modelos almacenados.

        Returns:
            Tupla (cantidad, primer_índice, último_índice). Si no hay modelos, devuelve (0, 0, 0).
        """
        if not self._indices:
            return 0, 0, 0
        return len(self._indices), self._indices[0], self._indices[-1]

    def delete_first(self, first: int) -> None:
        """
        Elimina el snapshot con el índice especificado (normalmente el más antiguo).

        Args:
            first: Índice del snapshot a eliminar.
        """
        (self.path / f"snapshotsSELECTION_{first}.pth").unlink(missing_ok=True)
        (self.path / f"snapshotsTURN_{first}.pth").unlink(missing_ok=True)
        self._player_cache.pop(first, None)   
        if first in self._indices:
            self.elos.pop(first, None)
            self._indices.remove(first)
        # FUTURO (Elo): self._elo_ratings.pop(first, None)

    def sample_assignment(self, N: int, pool_porcentage: float,agent_elo : float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Asigna un checkpoint de la pool a un subconjunto de partidas.

        Args:
            N: Número total de partidas en el lote.
            pool_porcentage: Fracción de partidas que usarán oponentes de la pool (0.0 - 1.0).

        Returns:
            from_pool: (N,) bool, True si la partida usa un oponente de la pool.
            checkpoint_idx: (N,) int, ID del checkpoint asignado (-1 si no usa pool).
        """
        cantidad, first_index, last_index = self.list_models()
        from_pool = torch.rand(N) < pool_porcentage

        if cantidad == 0:
            from_pool[:] = False

        checkpoint_idx = torch.full((N,), -1, dtype=torch.long)
        

        if cantidad > 0:
            n_from_pool = from_pool.sum().item()
            if n_from_pool > 0:
                elos_pool = torch.tensor([self.elos[idx] for idx in self._indices], dtype=torch.float)
                distancias = torch.abs(elos_pool - agent_elo)
                pesos_softmax = torch.softmax(-distancias / constants.ELO_TEMPERATURE, dim=0)
                posiciones = torch.multinomial(pesos_softmax, num_samples=n_from_pool, replacement=True)
                indices_tensor = torch.tensor(self._indices, dtype=torch.long)
                checkpoint_idx[from_pool] = indices_tensor[posiciones]

        return from_pool, checkpoint_idx

    def build_grouped_opponents(
        self,
        checkpoint_idx: torch.Tensor,
        player_class: Any,
        N: int,
        environment: Any,
    ) -> Dict[int, Tuple[Any, torch.Tensor]]:
        """
        Construye un diccionario agrupando partidas por checkpoint de oponente.

        Carga cada modelo una sola vez y lo reutiliza para todas las partidas
        que tengan asignado ese checkpoint.

        Args:
            checkpoint_idx: (N,) IDs de checkpoint asignados a cada partida.
            player_class: Clase del jugador (ej. PlayerAIV) para instanciar.
            N: Número total de partidas (para crear el jugador).
            environment: Instancia del entorno (necesaria para inicializar PlayerAIV).

        Returns:
            Diccionario {checkpoint_id: (instancia_jugador, índices_partidas)}.
        """
        grupos: Dict[int, Tuple[Any, torch.Tensor]] = {}
        unique_ids = checkpoint_idx.unique()

        for cp_id in unique_ids.tolist():
            if cp_id == -1:
                continue

            partida_indices = (checkpoint_idx == cp_id).nonzero(as_tuple=True)[0]

            if cp_id not in self._player_cache:
                jugador = player_class(N, environment,use_replay = False)
                path_sel = self.path / f"snapshotsSELECTION_{cp_id}.pth"
                path_turn = self.path / f"snapshotsTURN_{cp_id}.pth"
                jugador.load_model_inference_only(path_sel, path_turn)
                self._player_cache[cp_id] = jugador

            grupos[cp_id] = (self._player_cache[cp_id], partida_indices)
        #PARA QUE TODOS LOS OPONENTES DE LA POOL SEAN SIEMPRE DETERMINISTAS
        for jugador, _ in grupos.values():
            jugador.selection_network.eval()
            jugador.turn_network.eval()
        return grupos
    
    def update_elo(self,id, new_elo):
        self.elos[id] = new_elo
        
    def get_elo(self, cp_id: int) -> float:
        return self.elos.get(cp_id, constants.ELO_INITIAL)
