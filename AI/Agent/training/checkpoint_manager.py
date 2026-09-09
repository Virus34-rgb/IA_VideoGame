    
import torch

import constants


class checkpoint_manager:
    def __init__(self, selection_network,target_selection_network,optimizer_sel,replay_memory_sel,
                 turn_network,target_turn_network,optimizer_turn,replay_memory_turn):
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
            (self.selection_network, self.target_selection_network, self.optimizer_sel, self.replay_memory_sel, "epsilon_sel", "replayed_selection"),
            (self.turn_network, self.target_turn_network, self.optimizer_turn, self.replay_memory_turn, "epsilon_turn", "replayed_turn"),
        ]

    def save_model(self, path1: str, path2: str,replayed_attr,elo,eps_attr) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            torch.save({
                "dqn": net.state_dict(), "targetdqn": target_net.state_dict(), "optimizer": opt.state_dict(),
                "epsilon": eps_attr, "replayed": replayed_attr,
                "replay_memory": replay_memory.state_dict(), "elo": elo,
            }, path)

    #Al cargar un modelo se tiene que resetear el noise para reiniciar el weight y bias catcheado
    def load_model(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            target_net.load_state_dict(checkpoint["targetdqn"])
            opt.load_state_dict(checkpoint["optimizer"])
            setattr(self, eps_attr, checkpoint["epsilon"])
            setattr(self, replayed_attr, checkpoint["replayed"])
            replay_memory.load_state_dict(checkpoint["replay_memory"])
            self.elo = float(checkpoint.get("elo", constants.ELO_INITIAL))

    def load_model_inference_only(self, path1: str, path2: str) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            checkpoint = torch.load(path, weights_only=False)
            net.load_state_dict(checkpoint["dqn"])
            setattr(self, eps_attr, checkpoint["epsilon"])
            self.elo = float(checkpoint.get("elo", constants.ELO_INITIAL))

    def save_model_inference_only(self, path1: str, path2: str,elo,eps_attr) -> None:
        for path, (net, target_net, opt, replay_memory, eps_attr, replayed_attr) in zip((path1, path2), self._network_specs()):
            torch.save({"dqn": net.state_dict(), "epsilon": eps_attr, "elo": elo}, path)