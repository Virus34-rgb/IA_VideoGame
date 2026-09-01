"""
Módulo para normalizar observaciones del entorno.
"""
import torch
import constants


class ObservationV:
    #Ya no se usa
    def __init__(self, pl_types, pl_alive, pl_speed_norm, pl_health_norm,
                 pl_cooldowns, opp_life, opp_disposition, turn):
        self.pl_types = pl_types
        self.pl_alive = pl_alive
        self.pl_speed_norm = pl_speed_norm
        self.pl_health_norm = pl_health_norm
        self.pl_cooldowns = pl_cooldowns
        self.opp_life = opp_life
        self.opp_disposition = opp_disposition
        self.turn = turn

    @staticmethod
    def id_to_one_hot(warrior_id: int, warrior_quantity: int) -> list[int]:
        result = [0] * warrior_quantity # [0,....,0] de longitud warrior_quantity
        if warrior_id:
            result[warrior_id - 1] = 1 # pone a 1 el valor del identificador del guerrero (warrior_id) en la lista result
        return result

    @staticmethod
    def normalize_abilities(cooldowns_4: list[bool]) -> list[int]:
        encoded = [0, 0, 0, 0, 0, 0] # [0,....,0] de longitud 6
        for position, en_cooldown in enumerate(cooldowns_4):
            encoded[position] = 0 if en_cooldown else 1 # pone a 0 si está en cooldown, 1 si no
        return encoded

    @staticmethod
    def normalize_batch(
        pl_types: torch.Tensor,
        pl_alive: torch.Tensor,
        pl_speed_norm: torch.Tensor,
        pl_health_norm: torch.Tensor,
        pl_cooldowns: torch.Tensor,
        opp_life: torch.Tensor,
        opp_disposition: torch.Tensor,
        turn_norm: torch.Tensor,
        own_instance_abilities: torch.Tensor,   # NUEVO (N,3,4)
    ) -> torch.Tensor:
        N = pl_types.shape[0] #cantidad de partidas

        idx = (pl_types - 1).clamp(min=0) # tipos de guerrero normalizados a rango [0, WARRIOR_QUANTITY-1], para poder hacer one-hot
        one_hot_own = torch.nn.functional.one_hot(idx, num_classes=constants.WARRIOR_QUANTITY).float() 
        one_hot_own = one_hot_own * pl_alive.unsqueeze(-1).float() # enmascara los guerreros muertos (pone a 0 el one-hot de los guerreros muertos)

        #speed (N,3) normalizado a rango [0,1], 0 si el guerrero está muerto
        #Health (N,3) normalizado a rango [0,1], 0 si el guerrero está muerto
        speed = torch.where(pl_alive, pl_speed_norm, torch.zeros_like(pl_speed_norm)).unsqueeze(-1) # normaliza la velocidad de los guerreros vivos
        health = torch.where(pl_alive, pl_health_norm, torch.zeros_like(pl_health_norm)).unsqueeze(-1) # normaliza la salud de los guerreros vivos

        cd_usable = (pl_cooldowns == 0).float()   #mascara de habilidades disponibles (1 si está disponible, 0 si está en cooldown)
        #tensor (N,3,4) con 1 si la habilidad está disponible y el guerrero está vivo, 0 si está en cooldown o el guerrero está muerto
        cd_usable = torch.where(pl_alive.unsqueeze(-1), cd_usable, torch.zeros_like(cd_usable)) 
        
        extra_zeros = torch.zeros(N, 3, 2)
        #(N,3,13) = (N,3,6) + (N,3,1) + (N,3,1) + (N,3,2)
        propio = torch.cat([one_hot_own, speed, health, cd_usable, extra_zeros], dim=-1)
        propio = propio.flatten(start_dim=1) #(N, 3*13) = (N,39)

        idx_opp = (opp_disposition - 1).clamp(min=0)
        one_hot_opp = torch.nn.functional.one_hot(idx_opp, num_classes=constants.WARRIOR_QUANTITY).float()
        one_hot_opp = one_hot_opp * (opp_disposition > 0).unsqueeze(-1).float()
        one_hot_opp = one_hot_opp.flatten(start_dim=1) #(N, 3*WARRIOR_QUANTITY) = (N,15)

        pool_onehot = torch.nn.functional.one_hot(own_instance_abilities, num_classes=constants.MAX_POOL_SIZE).float()  # (N,3,4,POOL)
        pool_onehot = pool_onehot * pl_alive.view(N, 3, 1, 1).float()   # a 0 si el slot está muerto
        pool_onehot_flat = pool_onehot.flatten(start_dim=1)             # (N, 3*4*POOL)

        return torch.cat(
            [propio, opp_life, one_hot_opp, turn_norm.unsqueeze(-1), pool_onehot_flat], dim=-1
        )