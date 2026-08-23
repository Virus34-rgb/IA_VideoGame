
class WarriorData:
    def __init__(
        self,
        id,
        name,
        max_health,
        speed,
        ability1,ability2,ability3,ability4
    ):
        self.id = id
        self.name = name
        self.max_health = max_health
        self.speed = speed
        self.abilities = [ability1,ability2,ability3,ability4]