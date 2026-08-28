
from Antiguos.observation import Observation
from constants import WARRIOR_QUANTITY


class Choose_state:
    def __init__(self,
                 pl_disposition,
                 pl_warriors,
                 opp_initial_warrior,
                 opp_initialPosition
                 ):
        self.pl_disposition = pl_disposition
        self.pl_warriors = pl_warriors #ids
        self.opp_initial_warrior = opp_initial_warrior  
        self.opp_initialPosition = opp_initialPosition
        
    def encode_choose_state(self):
        observation = []
        for warrior in self.pl_disposition:
            if warrior == None:
                observation.extend([0] * WARRIOR_QUANTITY)
            else:
                observation.extend(
                    Observation.id_to_one_hot(
                        warrior.warrior_data.id,
                        WARRIOR_QUANTITY
                    )
                )
        for id in self.pl_warriors:
            observation.extend(Observation.id_to_one_hot(id, WARRIOR_QUANTITY))
        observation.extend(Observation.id_to_one_hot(self.opp_initial_warrior,WARRIOR_QUANTITY))
        observation.append(self.opp_initialPosition/3)
        return observation
        