"""
Gestión de guardado/carga de checkpoints para un agente PlayerAIV.

"""
import torch


class CheckpointManager:
    def __init__(self, selection_network, target_selection_network, optimizer_sel, replay_memory_sel,
                 turn_network, target_turn_network, optimizer_turn, replay_memory_turn):
        # Categoría A: mismas instancias que ya existían en PlayerAIV, no copias.
        self.selection_network = selection_network
        self.target_selection_network = target_selection_network
        self.optimizer_sel = optimizer_sel
        self.replay_memory_sel = replay_memory_sel
        self.turn_network = turn_network
        self.target_turn_network = target_turn_network
        self.optimizer_turn = optimizer_turn
        self.replay_memory_turn = replay_memory_turn

    def _network_specs(self):
        return [
            (self.selection_network, self.target_selection_network, self.optimizer_sel, self.replay_memory_sel),
            (self.turn_network, self.target_turn_network, self.optimizer_turn, self.replay_memory_turn),
        ]

    def save_model(self, path1: str, path2: str, epsilons: tuple, replayed_counts: tuple, elo: float) -> None:
        """epsilons=(epsilon_sel, epsilon_turn), replayed_counts=(replayed_selection, replayed_turn) -- VALORES."""
        for path, (net, target_net, opt, replay_memory), epsilon, replayed in zip(
            (path1, path2), self._network_specs(), epsilons, replayed_counts
        ):
            torch.save({
                "dqn": net.state_dict(), "targetdqn": target_net.state_dict(), "optimizer": opt.state_dict(),
                "epsilon": epsilon, "replayed": replayed,
                "replay_memory": replay_memory.state_dict(), "elo": elo,
            }, path)

    def load_model(self, path1: str, path2: str) -> dict:
        """Devuelve {"epsilons": (sel, turn), "replayed_counts": (sel, turn), "elo": float}.
        Quien llama debe asignar estos valores a sus propios atributos."""
        epsilons, replayed_counts, elo = [], [], None
        for path, (net, target_net, opt, replay_memory) in zip((path1, path2), self._network_specs()):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            target_net.load_state_dict(checkpoint["targetdqn"])
            opt.load_state_dict(checkpoint["optimizer"])
            replay_memory.load_state_dict(checkpoint["replay_memory"])
            epsilons.append(checkpoint["epsilon"])
            replayed_counts.append(checkpoint["replayed"])
            elo = float(checkpoint.get("elo", elo if elo is not None else 1000.0))
        return {"epsilons": tuple(epsilons), "replayed_counts": tuple(replayed_counts), "elo": elo}

    def load_model_inference_only(self, path1: str, path2: str) -> dict:
        epsilons, elo = [], None
        for path, (net, target_net, opt, replay_memory) in zip((path1, path2), self._network_specs()):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            epsilons.append(checkpoint["epsilon"])
            elo = float(checkpoint.get("elo", elo if elo is not None else 1000.0))
        return {"epsilons": tuple(epsilons), "elo": elo}

    def save_model_inference_only(self, path1: str, path2: str, epsilons: tuple, elo: float) -> None:
        for path, (net, target_net, opt, replay_memory), epsilon in zip((path1, path2), self._network_specs(), epsilons):
            torch.save({"dqn": net.state_dict(), "epsilon": epsilon, "elo": elo}, path)