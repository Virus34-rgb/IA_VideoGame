"""
Progresión del meta-juego de castillo tras cada batalla (envejecimiento,
muertes por edad, compra reactiva). Extraído de TrainerV: _run_meta_step,
_tipo_mas_repetido, _sample_categorical_shared, _traducir_muertes_combate.

Diseño: p1_castle/p2_castle son referencias A (ya existían en TrainerV,
propiedad conceptual del entrenador, pero la LÓGICA de cómo evolucionan sí es
responsabilidad de este colaborador). N vive aquí porque se usa en el muestreo.
"""
import torch

from AI.Meta.shop_heuristics import decidir_compra_batch
import constants


class CastleProgressionManager:
    def __init__(self, N: int, p1_castle, p2_castle):
        self.N = N
        self.p1_castle = p1_castle   # Categoría A: mismas instancias que TrainerV ya construyó.
        self.p2_castle = p2_castle

    def advance(self, castle_slots_p1, castle_slots_p2, p1_alive_final, p2_alive_final, stats):
        self.p1_castle.envejecer_heroes(castle_slots_p1)
        self.p2_castle.envejecer_heroes(castle_slots_p2)
        self.p1_castle.resolver_muertes(self._traducir_muertes_combate(castle_slots_p1, p1_alive_final))
        self.p2_castle.resolver_muertes(self._traducir_muertes_combate(castle_slots_p2, p2_alive_final))
        self.p1_castle.gold += constants.GOLD_POR_BATALLA
        self.p2_castle.gold += constants.GOLD_POR_BATALLA

        warrior_most_use_p1, warrior_most_use_p2 = self._tipo_mas_repetido(stats)
        for _ in range(constants.MAX_DEATHS_PER_TEAM):
            mask_compra_p1, tipo_p1 = decidir_compra_batch(self.p1_castle, warrior_most_use_p1)
            mask_compra_p2, tipo_p2 = decidir_compra_batch(self.p2_castle, warrior_most_use_p2)
            self.p1_castle.comprar_heroes(mask_compra_p1, tipo_p1)
            self.p2_castle.comprar_heroes(mask_compra_p2, tipo_p2)

    def _tipo_mas_repetido(self, stats):
        usage_p1 = stats._p1_warrior_use_ema
        usage_p2 = stats._p2_warrior_use_ema
        tipo_p1 = self._sample_categorical_shared(usage_p1)
        tipo_p2 = self._sample_categorical_shared(usage_p2)
        return tipo_p1, tipo_p2

    def _sample_categorical_shared(self, usage: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(usage / constants.SHOP_TEMPERATURE, dim=0)
        cumprobs = torch.cumsum(probs, dim=0)
        u = torch.rand(self.N)
        idx = torch.searchsorted(cumprobs, u).clamp(max=constants.WARRIOR_QUANTITY - 1)
        return idx + 1

    def _traducir_muertes_combate(self, castle_slots, alive_final):
        N = castle_slots.shape[0]
        max_size = constants.MAX_CASTLE_SIZE
        mask_muertes = torch.zeros((N, max_size), dtype=torch.bool)
        for slot in range(3):
            muertos = ~alive_final[:, slot]
            ids = castle_slots[muertos, slot]
            mask_muertes[muertos, ids] = True
        return mask_muertes