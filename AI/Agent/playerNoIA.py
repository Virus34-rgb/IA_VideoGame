
import random

import torch
from torch import nn
from AI.Agent.turnNetwork import TurnNetwork
from AI.Agent.observation import Observation
from AI.Agent.selectionNetwork import SelectionNetwork
from AI.Agent.replayMemoryAN import ReplayMemoryAN
from AI.Agent.replayMemoryPM import ReplayMemoryPM
from constants import ABILITIES, BATCH_SIZE, COPY_DQN, DISCOUNT_FACTOR, EPSILON_SEL_DECAY, EPSILON_SEL_MIN, EPSILON_SELECTION, EPSILON_TURN, EPSILON_TURN_DECAY, EPSILON_TURN_MIN, SELECTION_LEARNING_RATE, SELECTION_REPLAY_DATA, TURN_LEARNING_RATE, TURN_REPLAY_DATA, WARRIOR_QUANTITY

class PlayerNoAI:
    def __init__(self):
        self.name = "DqnPlayer"
        
    def selection(
        self,
        cs
    ):
        distribucion = ""
        pos_disponibles = []
        for pos,w in enumerate(cs.pl_disposition):
            if w is not None : distribucion += w.warrior_data.name
            else: 
                distribucion += "None"
                pos_disponibles.append(pos)
            distribucion += " "
        print(f"Tu distribucion:   {distribucion}\n")
        w_disponibles = []
        print(f"{cs.opp_initial_warrior}")
        if cs.opp_initial_warrior != 0:
            print(f"Enemigos seleccionado (el primero): {cs.pl_warriors[cs.opp_initial_warrior].name} en posicion {cs.opp_initialPosition}\n")
        for w in cs.pl_warriors.values():
            if(not w in cs.pl_disposition):
                w_disponibles.append(w)
        print(f"Personajes elegibles:   \n")
        for w in w_disponibles:
            print(f"{w.id}.{w.name} \n")
        ids_disponibles = [w.id for w in w_disponibles]
        elegido = int(input("Elige uno de los personajes elegibles (número antes del nombre)\n"))
        while elegido not in ids_disponibles:
            elegido = int(input("Elige uno de los personajes elegibles (número antes del nombre)\n"))
        print(f"Posiciones disponibles:")
        for pos in pos_disponibles:
            print(f" {pos} ")
        pos = int(input("\n Elige una de las posiciones disponibles\n"))
        while pos not in pos_disponibles:
            pos = int(input("\n Elige una de las posiciones disponibles\n"))
        warrior = next(
                w for w in w_disponibles
                if w.id == elegido
            )
        return warrior.id,pos,None
        
    
    def turn(self, observation):
        enemigos = {1:"Knight",2:"Archer",3:"Rogue",4:"Wizard",5:"Cleric"}
        print(f"Turno actual : {observation.turn}\n")
        print(f"Estado del rival:\n")
        for pos,enemy in enumerate(observation.opp_disposition):
            if(enemy!= None):
                print(f"{enemigos[enemy]} : Vida : {observation.opp_life[pos] * 100}%")
        actions = [None,None,None]
        for pos,w in enumerate(observation.pl_warriors) :
            if(w != None):
                ability_names = [(ability.name,ability.id) for ability in w.usable_abilities()]
                if(pos != 2):
                    ability_names.append(("Movimiento Positivo","MovPos"))
                if(pos != 0):
                    ability_names.append(("Movimiento Negativo","MovNeg"))
                print(f"{w.warrior_data.name}: Vida= {w.health}/{w.warrior_data.max_health}\n")
                print(f"Habilidades disponibles:\n")
                for name,id in ability_names:
                    print(f"      {id}.{name}")
                actions[pos] = int(input("Elige una habilidad disponible para este guerrero\n"))
                while not any(id_actual == actions[pos] for _, id_actual in ability_names):
                    actions[pos] = int(input("Elige una habilidad disponible para este guerrero\n"))
        return actions
    