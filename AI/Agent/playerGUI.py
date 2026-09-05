"""
Jugador humano con interfaz gráfica Tkinter.
Soporta modo Catálogo (5 tipos) y modo Castillo (MAX_CASTLE_SIZE slots).
"""
import tkinter as tk
import numpy as np
import torch

import constants


class PlayerGUIV:
    def __init__(self, environment):
        self.name = "PlayerGUIV"
        self.root = None
        self.environment = environment
        # self.warrior_names hardcodeado eliminado; se usa environment.warriors_classes
        # como única fuente de verdad (ver _warrior_name).

        # Variables para la fase de selección
        self.selection_done = False
        self.selected_item = None   # En catálogo: warrior_id (1..5). En castillo: slot_index (0..9)
        self.selected_position = None
        # Variables para la fase de turno
        self.turn_done = False
        self.actions = [None, None, None]
        self.confirm_btn = None
        self.elo = constants.ELO_INITIAL

    def _warrior_name(self, warrior_id: int) -> str:
        """Devuelve el nombre del guerrero a partir de su ID, o 'Vacío'/'Desconocido'."""
        if not warrior_id or warrior_id <= 0:
            return "Vacío"
        warrior_data = self.environment.warriors_classes.get(warrior_id)
        return warrior_data.name if warrior_data is not None else "Desconocido"

    # NUEVO: construye una etiqueta legible para el botón de una habilidad,
    # a partir del AbilityData real (no solo su índice). Sin esto, el jugador
    # humano no tenía forma de distinguir qué hacía cada botón de habilidad.
    _EFFECT_LABELS = {
        "ATTACK": "Ataque",
        "SELF_HEAL": "Autocura",
        "TEAM_HEAL": "Cura equipo",
        "DEFEND_FULL": "Defensa total",
        "DEFEND_HALF": "Defensa parcial",
    }

    def _ability_label(self, ability) -> str:
        tipo_efecto = self._EFFECT_LABELS.get(ability.effect_type.name, ability.effect_type.name)
        detalle = f"dmg/cura={ability.damage}" if ability.damage else "sin daño"
        cd_txt = f"cd={ability.turn_cd}" if ability.turn_cd > 0 else "sin cd"
        return f"{ability.name}\n({tipo_efecto}, {detalle}, {cd_txt})"

    # ------------------------------------------------------------
    # Máscara de acciones válidas — idéntica a PlayerAIV.compute_action_mask.
    # NUEVO: antes turn() decidía qué botones mostrar solo con cd[hab]==0 y
    # slot<2/slot>0, sin comprobar si una habilidad de ataque tiene algún
    # objetivo enemigo válido. Esto permitía mostrar botones que el entorno
    # luego trataría como "sin efecto" (target_mask sin coincidencia), dando
    # una experiencia distinta a la que ve la red. Duplicada aquí porque su
    # lógica no depende de ninguna red neuronal, solo de disposición/
    # cooldowns/vida/habilidades — el humano necesita la misma información.
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
        # NUEVO: no-op. No hay NoisyLinear en un jugador humano; se añade solo
        # por si algún punto del pipeline (p.ej. un guard futuro en TrainerV)
        # llega a invocarlo incondicionalmente sobre el oponente.
        pass

    # ---------- Métodos requeridos por el entrenador ----------
    def selection(self, batch_encoded_states, disposition, opp_initial_warrior,
                  castle_alive=None, already_used=None, castle_types=None):
        """
        Muestra la ventana para elegir:
          - Modo catálogo: un tipo (1..5) y una posición.
          - Modo castillo: un slot del castillo (0..9) y una posición.
        Retorna (item_index, position, action_dummy).
        """
        self._init_root()
        self.selection_done = False
        self.selected_item = None
        self.selected_position = None

        # Limpiar ventana
        for w in self.root.winfo_children():
            w.destroy()

        # Obtener disposición actual (N=1)
        disp = disposition.squeeze(0).tolist() if torch.is_tensor(disposition) else disposition
        opp = opp_initial_warrior.item() if torch.is_tensor(opp_initial_warrior) else opp_initial_warrior

        # Cabecera
        info = f"Tu disposición: {self._fmt_disp(disp)}\n"
        info += f"Enemigo seleccionó: {self._warrior_name(opp)}\n"
        info += "Elige un guerrero/slot y luego una posición libre."
        tk.Label(self.root, text=info, wraplength=400).pack(pady=10)

        # ---- Construir lista de elementos elegibles ----
        if constants.USE_META_GAME:
            # Modo CASTILLO: mostrar solo slots vivos y no usados
            alive = castle_alive.squeeze(0).tolist() if torch.is_tensor(castle_alive) else castle_alive
            used = already_used.squeeze(0).tolist() if torch.is_tensor(already_used) else already_used
            types = castle_types.squeeze(0).tolist() if torch.is_tensor(castle_types) else castle_types

            elegibles = []
            for slot in range(constants.MAX_CASTLE_SIZE):
                if alive[slot] and not used[slot]:
                    tipo = types[slot]
                    nombre = self._warrior_name(tipo)
                    elegibles.append((slot, nombre))
            
            if not elegibles:
                # Fallback: si no hay ninguno, coge el primero vivo (por si acaso)
                for slot in range(constants.MAX_CASTLE_SIZE):
                    if alive[slot]:
                        elegibles.append((slot, self._warrior_name(types[slot])))
                        break

            frame_items = tk.Frame(self.root)
            frame_items.pack(pady=5)
            tk.Label(frame_items, text="Slots del castillo disponibles:").pack()
            
            for slot, nombre in elegibles:
                btn = tk.Button(frame_items, text=f"Slot {slot}: {nombre}",
                                command=lambda s=slot: self._set_item(s))
                btn.pack(pady=2)
        else:
            # Modo CATÁLOGO: mostrar tipos no usados en la disposición actual
            disponibles = [w for w in range(1, 6) if w not in disp]
            if not disponibles:
                disponibles = [1]

            frame_items = tk.Frame(self.root)
            frame_items.pack(pady=5)
            tk.Label(frame_items, text="Guerreros disponibles:").pack()
            for w in disponibles:
                btn = tk.Button(frame_items, text=f"{w}: {self._warrior_name(w)}",
                                command=lambda wid=w: self._set_item(wid))
                btn.pack(side=tk.LEFT, padx=5)

        # ---- Posiciones libres (común a ambos modos) ----
        pos_libres = [i for i, v in enumerate(disp) if v == 0]
        if not pos_libres:
            pos_libres = [0]

        frame_pos = tk.Frame(self.root)
        frame_pos.pack(pady=5)
        tk.Label(frame_pos, text="Posiciones libres:").pack()
        for p in pos_libres:
            btn = tk.Button(frame_pos, text=f"Posición {p}",
                            command=lambda pos=p: self._set_position(pos))
            btn.pack(side=tk.LEFT, padx=5)

        # Esperar a que el usuario elija
        while not self.selection_done:
            self.root.update()
            self.root.update_idletasks()

        # En modo castillo, self.selected_item ya es el slot (0..9).
        # En modo catálogo, es el warrior_id (1..5).
        return self.selected_item, self.selected_position, 0

    def turn(self, batch_encoded_obs, own_disposition, own_cooldowns,
             own_alive, enemy_disposition, own_instance_abilities):
        """
        Muestra la ventana para elegir acciones.
        Extrae y muestra las vidas normalizadas del tensor de observación.
        Solo muestra botones para acciones que compute_action_mask marca
        como válidas (antes: solo se comprobaba cooldown y posición del slot).
        """
        self._init_root()
        self.turn_done = False
        self.actions = [None, None, None]
        self.confirm_btn = None

        # Limpiar ventana
        for w in self.root.winfo_children():
            w.destroy()

        # NUEVO: máscara real de acciones válidas (N,3,6), igual que ve la red.
        action_mask = self.compute_action_mask(
            own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities
        )

        # ------------------------------------------------------------
        # 1. EXTRAER VIDAS DEL TENSOR DE OBSERVACIÓN
        # ------------------------------------------------------------
        # batch_encoded_obs tiene forma (1, TURN_STATE_DIM)
        obs = batch_encoded_obs.squeeze(0).cpu().numpy()  # lo pasamos a numpy para manejarlo fácil
        
        # Índices según ObservationV.normalize_batch:
        # - Índices 18, 19, 20 -> Vida de tus guerreros (posiciones 0, 1, 2)
        # - Índices 39, 40, 41 -> Vida de los enemigos (posiciones 0, 1, 2)
        own_health_norm = np.array([obs[6], obs[19], obs[32]])
        enemy_health_norm = np.array([obs[39], obs[40], obs[41]])

        # Recortar por si hay algún valor fuera de rango (por seguridad)
        own_health_norm = np.clip(own_health_norm, 0.0, 1.0)
        enemy_health_norm = np.clip(enemy_health_norm, 0.0, 1.0)
        # ------------------------------------------------------------

        # Mostrar ENEMIGOS (con su vida)
        enemy_disp = enemy_disposition.squeeze(0).tolist()
        enemy_parts = []
        for i, t in enumerate(enemy_disp):
            if t != 0:
                hp_pct = int(enemy_health_norm[i] * 100)
                name = self._warrior_name(t)
                enemy_parts.append(f"{name} ({hp_pct}%)")
            else:
                enemy_parts.append("Vacío")
        enemy_txt = "Enemigos: " + " | ".join(enemy_parts)
        tk.Label(self.root, text=enemy_txt, font=("Arial", 10, "bold")).pack(pady=5)

        # Por cada slot propio
        for slot in range(3):
            alive = own_alive[0, slot].item() if torch.is_tensor(own_alive) else own_alive[0, slot]
            
            # Si el guerrero está muerto, marcamos acción -1 y no mostramos botones
            if not alive:
                self.actions[slot] = -1
                # Aun así mostramos un frame gris para mantener el orden visual
                frame = tk.Frame(self.root, relief=tk.RAISED, borderwidth=2, bg="lightgray")
                frame.pack(pady=5, fill=tk.X)
                tk.Label(frame, text=f"Posición {slot}: MUERTO", bg="lightgray").pack()
                continue

            frame = tk.Frame(self.root, relief=tk.RAISED, borderwidth=2)
            frame.pack(pady=5, fill=tk.X)

            tipo = own_disposition[0, slot].item()
            hp_pct = int(own_health_norm[slot] * 100)
            
            # Mostrar nombre y vida
            label = tk.Label(frame, text=f"Posición {slot}: {self._warrior_name(tipo)} - ❤️ {hp_pct}% HP")
            label.pack()

            # Cooldowns (informativo, se mantiene igual)
            cd = own_cooldowns[0, slot].tolist()
            cd_txt = "Cooldowns: " + " | ".join(f"Hab{i}:{c}" for i, c in enumerate(cd))
            tk.Label(frame, text=cd_txt).pack()

            own_abilities_slot = own_instance_abilities[0, slot].tolist()  # (4,) índices de pool
            for hab in range(4):
                if action_mask[0, slot, hab]:
                    pool_idx = own_abilities_slot[hab]
                    ability = self.environment.warriors_classes[tipo].ability_pool[pool_idx]
                    etiqueta = self._ability_label(ability)
                    btn = tk.Button(frame, text=etiqueta,
                                    command=lambda s=slot, h=hab: self._set_action(s, h))
                    btn.pack(side=tk.LEFT, padx=2)

            # CAMBIADO: antes `if slot < 2` / `if slot > 0` — ahora se usa
            # action_mask[0, slot, 4]/[0, slot, 5], que ya codifica exactamente
            # las mismas restricciones de borde (mask[:,0,5]=False, mask[:,2,4]
            # =False) desde el único punto de verdad compartido con la red.
            if action_mask[0, slot, 5]:
                btn = tk.Button(frame, text="→ Mover derecha",
                                command=lambda s=slot: self._set_action(s, 5))
                btn.pack(side=tk.LEFT, padx=2)
            if action_mask[0, slot, 4]:
                btn = tk.Button(frame, text="← Mover izquierda",
                                command=lambda s=slot: self._set_action(s, 6))
                btn.pack(side=tk.LEFT, padx=2)

        # Botón de confirmación
        self.confirm_btn = tk.Button(self.root, text="✅ Confirmar turno",
                                     command=self._confirm, state=tk.DISABLED)
        self.confirm_btn.pack(pady=10)

        # Bucle de espera (la GUI se mantiene viva)
        while not self.turn_done:
            self.root.update()
            self.root.update_idletasks()

        return torch.tensor(self.actions, dtype=torch.long).unsqueeze(0)

    # ---------- Métodos internos ----------
    def _init_root(self):
        if self.root is None:
            self.root = tk.Tk()
            self.root.title("Castle Game - Jugador Humano")
            self.root.geometry("500x550")

    def _set_item(self, item):
        self.selected_item = item
        self._check_selection()

    def _set_position(self, pos):
        self.selected_position = pos
        self._check_selection()

    def _check_selection(self):
        if self.selected_item is not None and self.selected_position is not None:
            self.selection_done = True

    def _set_action(self, slot, action):
        self.actions[slot] = action
        if all(a is not None for a in self.actions):
            self.confirm_btn.config(state=tk.NORMAL)

    def _confirm(self):
        self.turn_done = True

    def _fmt_disp(self, disp):
        return " | ".join(self._warrior_name(v) for v in disp)