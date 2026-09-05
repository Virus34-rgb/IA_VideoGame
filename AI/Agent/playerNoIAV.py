"""
Jugador humano interactivo para Castle Game.

Reescrito para ser compatible con el pipeline vectorizado actual (tensores,
draft de castillo). Asume N=1 (una única partida interactiva). No usa redes
neuronales, no tiene replay memory, y no participa en el pool de oponentes
ni en el sistema de Elo (esos mecanismos se desactivan para este modo en
TrainerV._run, ver guard (learn_p1 or learn_p2)).
"""
import torch

import constants


class PlayerNoAIV:
    def __init__(self, environment):
        self.name = "PlayerNoAIV"
        self.environment = environment
        # Inofensivo: no se actualiza nunca, solo evita AttributeError si algún
        # código lo consulta de forma incondicional.
        self.elo = constants.ELO_INITIAL

    # ------------------------------------------------------------
    # Draft (selección de equipo, modo castillo)
    # ------------------------------------------------------------
    def selection(self, batch_encoded_states, disposition, opp_initial_warrior, castle_alive=None, already_used=None, castle_types=None):
        if not constants.USE_META_GAME:
            raise NotImplementedError("PlayerNoAIV solo soporta el modo castillo (USE_META_GAME=True).")

        print("\n" + "=" * 60)
        print("FASE DE SELECCIÓN")
        print("=" * 60)

        disp0 = disposition[0]
        print("Tu equipo actual:")
        for pos in range(3):
            tipo = disp0[pos].item()
            if tipo > 0:
                print(f"  Posición {pos}: {self.environment.warriors_classes[tipo].name}")
            else:
                print(f"  Posición {pos}: (vacío)")

        opp_w = opp_initial_warrior[0].item() if isinstance(opp_initial_warrior, torch.Tensor) else opp_initial_warrior
        if opp_w and opp_w > 0:
            print(f"\nEl rival empezó con: {self.environment.warriors_classes[opp_w].name}")

        disponibles = castle_alive[0] & ~already_used[0]
        print("\nGuerreros disponibles en tu castillo:")
        slots_validos = []
        for slot in range(constants.MAX_CASTLE_SIZE):
            if disponibles[slot]:
                tipo = castle_types[0, slot].item()
                nombre = self.environment.warriors_classes[tipo].name
                print(f"  Slot {slot}: {nombre}")
                slots_validos.append(slot)

        if not slots_validos:
            raise RuntimeError("No hay guerreros disponibles en el castillo para seleccionar (draft agotado).")

        slot_elegido = int(input(f"\nElige un slot ({slots_validos}): "))
        while slot_elegido not in slots_validos:
            slot_elegido = int(input(f"Slot inválido. Elige un slot ({slots_validos}): "))

        posiciones_libres = [pos for pos in range(3) if disp0[pos].item() == 0]
        pos_elegida = int(input(f"Elige una posición de combate ({posiciones_libres}): "))
        while pos_elegida not in posiciones_libres:
            pos_elegida = int(input(f"Posición inválida. Elige una posición ({posiciones_libres}): "))

        action = slot_elegido * 3 + pos_elegida
        return (
            torch.tensor([slot_elegido], dtype=torch.long),
            torch.tensor([pos_elegida], dtype=torch.long),
            torch.tensor([action], dtype=torch.long),
        )

    # ------------------------------------------------------------
    # Turno de combate
    # ------------------------------------------------------------
    def turn(self, batch_encoded_obs, own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities):
        action_mask = self.compute_action_mask(own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities)

        # own_disposition es literalmente el mismo tensor que environment.p1_disposition
        # o environment.p2_disposition en el momento de esta llamada (ver _run_turn /
        # _turn_mixed_opponent en TrainerV) — la comparación de identidad permite saber
        # de qué lado es este jugador sin necesitar un parámetro extra.
        soy_p1 = own_disposition is self.environment.p1_disposition
        own_health = self.environment.p1_healths if soy_p1 else self.environment.p2_healths
        enemy_health = self.environment.p2_healths if soy_p1 else self.environment.p1_healths

        print("\n" + "-" * 60)
        print(f"TURNO {self.environment.turn_number[0].item()}")
        print("-" * 60)

        print("Estado del rival:")
        for pos in range(3):
            tipo = enemy_disposition[0, pos].item()
            if tipo > 0:
                maxh = self.environment.max_health_por_tipo[tipo].item()
                vida_pct = (enemy_health[0, pos].item() / maxh * 100) if maxh > 0 else 0.0
                print(f"  Posición {pos}: {self.environment.warriors_classes[tipo].name} — {vida_pct:.0f}% vida")

        actions = [-1, -1, -1]
        for pos in range(3):
            if not own_alive[0, pos]:
                continue

            tipo = own_disposition[0, pos].item()
            warrior_data = self.environment.warriors_classes[tipo]
            maxh = warrior_data.max_health
            vida_pct = (own_health[0, pos].item() / maxh * 100) if maxh > 0 else 0.0
            print(f"\n{warrior_data.name} (posición {pos}) — {vida_pct:.0f}% vida")

            opciones = []
            for boton in range(4):
                if action_mask[0, pos, boton]:
                    pool_idx = own_instance_abilities[0, pos, boton].item()
                    nombre_habilidad = warrior_data.ability_pool[pool_idx].name
                    opciones.append((boton, nombre_habilidad))
            if action_mask[0, pos, 4]:
                opciones.append((5, "Movimiento Positivo"))   # código de entorno 5
            if action_mask[0, pos, 5]:
                opciones.append((6, "Movimiento Negativo"))   # código de entorno 6

            if not opciones:
                print("  Sin acciones disponibles este turno.")
                continue

            print("  Acciones disponibles:")
            for codigo, nombre in opciones:
                print(f"    {codigo}. {nombre}")

            codigos_validos = [codigo for codigo, _ in opciones]
            elegido = int(input("  Elige una acción: "))
            while elegido not in codigos_validos:
                elegido = int(input("  Acción inválida. Elige una acción: "))
            actions[pos] = elegido

        return torch.tensor([actions], dtype=torch.long)

    # ------------------------------------------------------------
    # Máscara de acciones válidas — idéntica a PlayerAIV.compute_action_mask.
    # Duplicada aquí porque su lógica no depende de ninguna red neuronal, solo
    # de disposición/cooldowns/vida/habilidades, así que un jugador humano
    # necesita exactamente la misma información para saber qué puede hacer.
    # ------------------------------------------------------------
    def compute_action_mask(self, own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities):
        N = own_disposition.shape[0]
        mask = own_alive.unsqueeze(-1).expand(N, 3, 6).clone()

        mask[:, :, :4] &= (own_cooldowns == 0)

        table = self.environment.target_mask_por_tipo_habilidad
        target_mask_pool = table[own_disposition]
        idx = own_instance_abilities.unsqueeze(-1).expand(-1, -1, -1, 3)
        target_mask_full = target_mask_pool.gather(2, idx)

        enemy_ocupado = (enemy_disposition > 0).unsqueeze(1).unsqueeze(1)
        hay_target_valido = (target_mask_full & enemy_ocupado).any(dim=-1)
        sin_target = ~hay_target_valido & target_mask_full.any(dim=-1)

        mask[:, :, :4] &= ~sin_target

        mask[:, 0, 5] = False
        mask[:, 2, 4] = False

        return mask

    def reset_noise(self):
        pass  # no aplica: no hay NoisyLinear en un jugador humano