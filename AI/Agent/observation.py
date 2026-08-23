from constants import MAX_TURNS, WARRIOR_QUANTITY


class Observation:
    def __init__(
        self,
        pl_warriors,
        opp_life,
        opp_disposition,
        turn,
    ):
        self.pl_warriors = pl_warriors
        self.opp_life = opp_life
        self.opp_disposition = opp_disposition
        self.turn = turn
        
    def normalize(self):
        observation = []
        # Mis guerreros
        for w in self.pl_warriors:
            if(w is None):
                observation.extend(self.id_to_one_hot(0,WARRIOR_QUANTITY))
                observation.append(0 / 20)
                observation.append(0)
                observation.extend([0,0,0,0,0,0])
            else:
                observation.extend(self.id_to_one_hot(w.warrior_data.id, WARRIOR_QUANTITY))
                observation.append(w.warrior_data.speed/ 20)
                observation.append(w.health / w.warrior_data.max_health)
                observation.extend(self.normalize_abilities(w))
        # Vida de los enemigos
        observation.extend(self.opp_life)
        # Posición/tipo de los enemigos
        for warrior_id in self.opp_disposition:
            observation.extend(
                self.id_to_one_hot(warrior_id, WARRIOR_QUANTITY)
            )
        # Turno
        observation.append(min(self.turn / MAX_TURNS, 1.0))
        return observation
    @staticmethod
    def id_to_one_hot(warrior_id, warrior_quantity):
        result = [0] * warrior_quantity
        if warrior_id:  # None o 0 → nada activado
            result[warrior_id - 1] = 1
        return result
    @staticmethod
    def normalize_abilities(warrior):
        encoded = [0,0,0,0,0,0]
        cooldown = warrior.cooldown_abilities
        for position,value in enumerate(cooldown):
            if(value == True):
                encoded[position] = 0
            else:
                encoded[position] = 1
        encoded[4] = 0 # añadimos las otras 3 posibilidades para que esten y luego las ponemos a 1 o 0 con la mascara 
        encoded[5] = 0 
        return encoded
                