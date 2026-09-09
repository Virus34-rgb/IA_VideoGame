

@staticmethod
def compute_action_mask(own_disposition, own_cooldowns, own_alive, enemy_disposition, own_instance_abilities,target_mask_table):
    """
    Calcula la máscara booleana de acciones válidas (N, 3, 6), sin aplicarla a
    ningún logit. Separado de mask_turn para poder calcularla una única vez en
    el momento de recolección y reutilizarla desde el replay buffer, en vez de
    recalcularla en cada sample de replay_turn/_multi_agent_double_dqn_target.
    """
    N = own_disposition.shape[0] #cantidad de partidas
    mask = own_alive.unsqueeze(-1).expand(N, 3, 6).clone() # (N, 3, 6) bool, inicialmente todas las acciones son válidas para guerreros vivos

    mask[:, :, :4] &= (own_cooldowns == 0) # si los cooldowns son mayores que 0, se deshabilitan las acciones de ataque (0-3)

    table = target_mask_table          # (num_types, POOL, 3)
    #Para cada partida, para cada guerrero, obtenemos la máscara de objetivos válidos según el tipo de habilidad del guerrero
    target_mask_pool = table[own_disposition]                        # (N, 3, POOL, 3)
    idx = own_instance_abilities.unsqueeze(-1).expand(-1, -1, -1, 3)  # (N, 3, 4, 3)
    #para cada geurrero de cada paritda, obtemeos de own_instance_abilities el indice de la habilidad que tiene, 
    # y con ese indice obtenemos de target_mask_pool la máscara de objetivos válidos para esa habilidad
    #target_mask_pool [:,:,idx,:]
    target_mask_full = target_mask_pool.gather(2, idx)                # (N, 3, 4, 3) PARTIDAS/POSICION/HABILIDAD/OBJETIVO
    
    #enemy_disposition es (N,3) con los tipos de guerreros enemigos (0 para muertos, 1..5 para vivos)
    #Despues (N,3,1,1) para poder compararlo con target_mask_full
    enemy_ocupado = (enemy_disposition > 0).unsqueeze(1).unsqueeze(1)
    #Si alguno es true, significa que hay al menos un objetivo válido para esa habilidad y guerrero
    hay_target_valido = (target_mask_full & enemy_ocupado).any(dim=-1)
    #no hay target válido si no hay ningún objetivo válido para esa habilidad y guerrero
    sin_target = ~hay_target_valido & target_mask_full.any(dim=-1)

    mask[:, :, :4] &= ~sin_target

    mask[:, 0, 5] = False # Slot 0 (front) → no puede moverse a la derecha (acción 5)
    mask[:, 2, 4] = False  # Slot 2 (back)  → no puede moverse a la izquierda (acción 4)

    return mask   # (N, 3, 6) bool