class Warrior:
    def __init__(self,warrior_data):
        self.warrior_data = warrior_data
        self.health = warrior_data.max_health
        self.cooldown_abilities = [False,False,False,False] #4ataques + 2 Movimientos
        
    def use_ability(self,pos):
        if(self.warrior_data.abilities[pos].can_repeat == False):
            self.cooldown_abilities[pos] = True
            
    def usable_abilities(self):
        usable = []
        for position,ability in enumerate(self.warrior_data.abilities):
            if(self.cooldown_abilities[position] == False):
                usable.append(ability)
        return usable #Devolvería una lista de abilities
    
    def reset_cooldowns(self):
        for position,cd in enumerate(self.cooldown_abilities):
            if cd is True:
                self.cooldown_abilities[position] = False
                
    def modify_health(self,damage,curation):
        self.health = min(self.warrior_data.max_health,self.health - damage + curation)
        
    def reset_health(self):
        self.health = self.warrior_data.max_health