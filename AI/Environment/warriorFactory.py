# warriorFactory.py


from AI.Environment.abilityData import AbilityData
from AI.Environment.warriorData import WarriorData


def get_warriors_classes():
    knight = WarriorData(
        id = 1,
        name="Knight",
        max_health= 30,
        speed = 10,
        ability1 = AbilityData("Smite",1,8,[0],False) ,
        ability2 = AbilityData("Guard Up",2,0,[],False),#inhabilita el daño a el en un turno
        ability3 = AbilityData("Slice",3,3,[0,1,2],True),
        ability4 = AbilityData("Throw",4,7,[2],True)
    )
    archer = WarriorData(
        id = 2,
        name="Archer",
        max_health= 20,
        speed = 18,
        ability1 = AbilityData("Arrow",1,6,[1],True) ,
        ability2 = AbilityData("Heal",2,6,[],False), #Se cura 5 de vida
        ability3 = AbilityData("Arrow2",3,6,[2],True),
        ability4 = AbilityData("Rain",4,5,[0,1,2],False)
    )
    rogue = WarriorData(
        id = 3,
        name = "Rogue",
        max_health= 10,
        speed = 20,
        ability1 = AbilityData("BackAttack",1,9,[2],False) ,
        ability2 = AbilityData("Hide",2,0,[],False), #Se cubre de daño 1 turno
        ability3 = AbilityData("PoisonGas",3,3,[0,1,2],False),
        ability4 = AbilityData("Knife",4,5,[0],True)
    )
    wizard = WarriorData(
        id = 4,
        name = "Wizard",
        max_health= 15,
        speed = 12,
        ability1 = AbilityData("Magic Missile",1,4,[0,1,2],True) ,
        ability2 = AbilityData("Zap",2,6,[1,2],False), 
        ability3 = AbilityData("FireBall",3,6,[0,1,2],False),
        ability4 = AbilityData("StaffAttack",4,4,[0],True)
    )
    cleric = WarriorData(
        id = 5,
        name = "Cleric",
        max_health= 20,
        speed = 14,
        ability1 = AbilityData("Charge",1,3,[0,1],True) ,
        ability2 = AbilityData("HealAll",2,4,[],False), #Cura a todos 5
        ability3 = AbilityData("Defend",3,4,[],False), #Reduce su daño recibido en un 50%
        ability4 = AbilityData("Light",4,4,[0,1,2],False)
    )
    warrior_classes = {knight.id:knight,archer.id:archer,rogue.id:rogue,wizard.id:wizard,cleric.id:cleric}
    return warrior_classes
    