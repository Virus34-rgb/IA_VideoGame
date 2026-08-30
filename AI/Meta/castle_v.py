import torch

from AI.Environment.abilitySampling import sample_abilities_batch
import constants


class CastleV:
    def __init__(self, N):
        self.N = N
        self.row_idx = torch.arange(N)

        self.castle_types = torch.zeros((N, constants.MAX_CASTLE_SIZE), dtype=torch.long)
        self.castle_abilities = torch.zeros((N, constants.MAX_CASTLE_SIZE, constants.ABILITIES_PER_WARRIOR), dtype=torch.long)
        self.castle_abilities_levels = torch.full((N, constants.MAX_CASTLE_SIZE, constants.ABILITIES_PER_WARRIOR), 1, dtype=torch.long)
        self.battle_fought = torch.zeros((N, constants.MAX_CASTLE_SIZE), dtype=torch.long)
        self.castle_alive = torch.zeros((N, constants.MAX_CASTLE_SIZE), dtype=torch.bool)
        self.castle_has_revival = torch.zeros((N, constants.MAX_CASTLE_SIZE), dtype=torch.bool)
        self.gold = torch.full((N,), 250, dtype=torch.long)
        self.inicializar()

    def reset(self):
        self.__init__(self.N)

    def comprar_heroes(self, mask_compra, tipo_elegido):
        hay_hueco = (~self.castle_alive).any(dim=1)
        mask_compra = mask_compra & hay_hueco

        slot_libre = (~self.castle_alive).float().argmax(dim=1) 

        new_type = torch.where(mask_compra, tipo_elegido, self.castle_types[self.row_idx, slot_libre])
        self.castle_types[self.row_idx, slot_libre] = new_type   

        self.castle_alive[self.row_idx, slot_libre] = mask_compra | self.castle_alive[self.row_idx, slot_libre]  

        abilities = sample_abilities_batch(constants.MAX_POOL_SIZE, self.N, constants.ABILITIES_PER_WARRIOR)
        new_abilities = torch.where(
            mask_compra.unsqueeze(-1), abilities, self.castle_abilities[self.row_idx, slot_libre] 
        )
        self.castle_abilities[self.row_idx, slot_libre] = new_abilities

        levels_default = torch.ones_like(self.castle_abilities_levels[self.row_idx, slot_libre])
        new_levels = torch.where(
            mask_compra.unsqueeze(-1), levels_default, self.castle_abilities_levels[self.row_idx, slot_libre]  
        )
        self.castle_abilities_levels[self.row_idx, slot_libre] = new_levels

        new_battles = torch.where(mask_compra, torch.zeros_like(self.battle_fought[self.row_idx, slot_libre]), self.battle_fought[self.row_idx, slot_libre])  # CORREGIDO
        self.battle_fought[self.row_idx, slot_libre] = new_battles

        self.gold = torch.where(mask_compra, self.gold - constants.COST_COMPRA, self.gold)

    def envejecer_heroes(self, idx_hero_slot):
        row_idx_expanded = self.row_idx.unsqueeze(1).expand(-1, idx_hero_slot.shape[1]) 
        self.battle_fought[row_idx_expanded, idx_hero_slot] += 1
        self.resolver_muertes_envejecimiento()  

    def resolver_muertes_envejecimiento(self):
        muertes = (self.battle_fought > constants.MAX_BATALLAS) & ~self.castle_has_revival 
        self.resolver_muertes(muertes)

    # Usado para resolver muertes por partida
    def resolver_muertes(self, mask_muertes):
        self.castle_alive = torch.where(mask_muertes, False, self.castle_alive)
        
    def inicializar(self):
        mask_compra = torch.ones(self.N,dtype= torch.bool)
        self.comprar_heroes(mask_compra,torch.full((self.N,), 1, dtype=torch.long))
        self.comprar_heroes(mask_compra,torch.full((self.N,), 2, dtype=torch.long))
        self.comprar_heroes(mask_compra,torch.full((self.N,), 3, dtype=torch.long))
        self.comprar_heroes(mask_compra,torch.full((self.N,), 4, dtype=torch.long))
        self.comprar_heroes(mask_compra,torch.full((self.N,), 5, dtype=torch.long))
        
    def state_dict(self):
        return {
            "castle_types": self.castle_types,
            "castle_abilities": self.castle_abilities,
            "castle_abilities_levels": self.castle_abilities_levels,
            "battle_fought": self.battle_fought,
            "castle_alive": self.castle_alive,
            "castle_has_revival": self.castle_has_revival,
            "gold": self.gold,
        }

    def load_state_dict(self, state):
        self.castle_types.copy_(state["castle_types"])
        self.castle_abilities.copy_(state["castle_abilities"])
        self.castle_abilities_levels.copy_(state["castle_abilities_levels"])
        self.battle_fought.copy_(state["battle_fought"])
        self.castle_alive.copy_(state["castle_alive"])
        self.castle_has_revival.copy_(state["castle_has_revival"])
        self.gold.copy_(state["gold"])