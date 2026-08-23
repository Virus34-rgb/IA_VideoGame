
class AbilityData:
    def __init__(self,name,id,damage,target_positions,can_repeat):
        self.name = name
        self.id = id #1-4
        self.damage = damage
        self.target_positions = target_positions
        self.can_repeat = can_repeat