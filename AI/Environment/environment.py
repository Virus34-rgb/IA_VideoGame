

from AI.Environment.gameState import GameState
from AI.Environment.stats import Stats
from AI.Environment.warrior import Warrior
from AI.Environment.warriorFactory import get_warriors_classes
from constants import DISCOUNT_FACTOR_TURN, REWARD_WEIGHTS, TURN_PENALTY, WIN_REWARD, MAX_TURNS, MAX_DEATHS_PER_TEAM

class Environment:
    def __init__ (
        self,
    ):
        self.warriors_classes = get_warriors_classes()
        self.p1_disposition = [None,None,None]
        self.p2_disposition = [None,None,None]
        self.p1_initialWarrior = 0 
        self.p2_initialWarrior = 0
        self.p1_initialPosition = 0
        self.p2_initialPosition = 0
        self.p1_deaths = 0
        self.p2_deaths = 0
        self.ended = False
        self.winner = None
        self.turn_number = 0
        self.stats = Stats()
    

    def reset (self):
        self.p1_disposition = [None,None,None]
        self.p2_disposition = [None,None,None]
        self.p1_initialWarrior = 0
        self.p2_initialWarrior = 0
        self.p1_initialPosition = 0
        self.p2_initialPosition = 0
        self.p1_deaths = 0
        self.p2_deaths = 0
        self.ended = False
        self.winner = None
        self.turn_number = 0
        return self.get_state()
    
    def team_selection(self,warrior_p1,pos1,warrior_p2,pos2):
        self.warrior_selected(warrior_p1,pos1,warrior_p2,pos2,0)
        return self.get_state()
    
    def warrior_selected(self,warrior1,pos1,warrior2,pos2,selected):
        if(selected == 0):
            self.p1_initialWarrior = warrior1
            self.p2_initialWarrior = warrior2
            self.p1_initialPosition = pos1 + 1
            self.p2_initialPosition = pos2 + 1
        self.p1_disposition[pos1] = Warrior(self.warriors_classes[warrior1])
        self.p2_disposition[pos2] = Warrior(self.warriors_classes[warrior2])
        self.stats.p1_warrior_use[warrior1 - 1] +=1
        self.stats.p2_warrior_use[warrior2 - 1] +=1
    
    def turn(self, actionsp1, actionsp2):
        self.turn_number += 1
        order = self.get_turn_order()
        damage_p1 = damage_p2 = 0
        damage_avoided_p1 = damage_avoided_p2 = 0
        blocks_p1 = blocks_p2 = 0
        heal_p1 = heal_p2 = 0
        p1_health = 0
        p2_health = 0
        for warr in self.p1_disposition:
            if(warr != None):
                p1_health += warr.health/ warr.warrior_data.max_health
        for warr in self.p2_disposition:
            if(warr != None):
                p2_health += warr.health/ warr.warrior_data.max_health
        for player, warr in order:
            if warr is None or warr.health <= 0:
                continue
            if player == 1:
                dmg, avoided, blocks, moved,heal = self._resolve_action(
                    warr, self.p1_disposition, self.p2_disposition,
                    actionsp1, actionsp2, self.stats.p1_attacks
                )
                heal_p1 += heal
                damage_p1 += dmg
                damage_avoided_p2 += avoided
                blocks_p2 += blocks
                self.stats.p1_movements += moved
            else:
                dmg, avoided, blocks, moved,heal = self._resolve_action(
                    warr, self.p2_disposition, self.p1_disposition,
                    actionsp2, actionsp1, self.stats.p2_attacks
                )
                heal_p2 += heal
                damage_p2 += dmg
                damage_avoided_p1 += avoided
                blocks_p1 += blocks
                self.stats.p2_movements += moved
        p1_post_health = 0
        p2_post_health = 0 
        for warr in self.p1_disposition:
            if(warr != None):
                p1_post_health += warr.health/ warr.warrior_data.max_health
        for warr in self.p2_disposition:
            if(warr != None):
                p2_post_health += warr.health/ warr.warrior_data.max_health
        shapping1 = p1_health/3 - p2_health/3
        shapping2 = p1_post_health/3 - p2_post_health/3
        rewardP1, rewardP2 = self.calculate_rewards(
            damage_p1, damage_p2, damage_avoided_p1, damage_avoided_p2,heal_p1,heal_p2,shapping1,shapping2
        )
        self.stats.p1_damage += damage_p1
        self.stats.p2_damage += damage_p2
        self.stats.p1_succes_blocks += blocks_p1
        self.stats.p2_succes_blocks += blocks_p2
        self.stats.p1_tot_damage_evaded += damage_avoided_p1
        self.stats.p2_tot_damage_evaded += damage_avoided_p2
        self.stats.p1_tot_heal += heal_p1
        self.stats.p2_tot_heal += heal_p2
        self.stats.total_reward_p1 += rewardP1
        self.stats.total_reward_p2 += rewardP2
        return self.get_state(), rewardP1, rewardP2, self.ended
    
    def _resolve_action(self, warr, own_disposition, enemy_disposition,own_actions, enemy_actions, attack_stats):
        pos = own_disposition.index(warr)
        action = own_actions[pos]
        if action in (1, 2, 3, 4):
            attack_stats[warr.warrior_data.id][action - 1] += 1
        if action == "movPos":
            moved = 0
            if pos != 2:
                moved = 1
                own_disposition[pos], own_disposition[pos + 1] = own_disposition[pos + 1], own_disposition[pos]
            return 0, 0, 0, moved,0
        if action == "movNeg":
            moved = 0
            if pos != 0:
                moved = 1
                own_disposition[pos], own_disposition[pos - 1] = own_disposition[pos - 1], own_disposition[pos]
            return 0, 0, 0, moved,0
        warr.reset_cooldowns()
        warr.use_ability(action - 1)
        if warr.warrior_data.id == 2 and action == 2:
            vida_ant = warr.health
            warr.modify_health(0, warr.warrior_data.abilities[action - 1].damage)
            vida_desp = warr.health
            return 0, 0, 0, 0,vida_desp-vida_ant
        if warr.warrior_data.id == 5 and action == 2:
            tot_health = 0
            for warrior in own_disposition:
                if warrior is not None:
                    vida_ant = warrior.health
                    warrior.modify_health(0, warr.warrior_data.abilities[action - 1].damage)
                    vida_desp = warrior.health
                    tot_health += vida_desp - vida_ant  
            return 0, 0, 0, 0, tot_health
        damage = 0
        damage_avoided = 0
        blocks = 0
        target_pos = warr.warrior_data.abilities[action - 1].target_positions
        for target in target_pos:
            enemy = enemy_disposition[target]
            if enemy is None:
                continue
            pos_enemy = enemy_disposition.index(enemy)
            action_enemy = enemy_actions[pos_enemy]
            enemy_id = enemy.warrior_data.id
            would_be_damage = warr.warrior_data.abilities[action - 1].damage

            if (enemy_id == 1 and action_enemy == 2) or (enemy_id == 3 and action_enemy == 2):
                hit_damage = 0
                damage_avoided += would_be_damage
                blocks += 1
            elif enemy_id == 5 and action_enemy == 3:
                hit_damage = would_be_damage / 2
                damage_avoided += would_be_damage / 2
                blocks += 1
            else:
                hit_damage = would_be_damage

            enemy.modify_health(hit_damage, 0)
            damage += hit_damage

        return damage, damage_avoided, blocks, 0,0

    def get_state(self) :
        return GameState(self.p1_disposition,
                              self.p2_disposition,
                              self.p1_deaths,
                              self.p2_deaths,
                              self.turn_number)
    def get_turn_order(self):
        order = []
        for pos in range(3):
            warrior = self.p1_disposition[pos]
            if warrior is not None:
                order.append((1, warrior, warrior.warrior_data.speed))

        for pos in range(3):
            warrior = self.p2_disposition[pos]
            if warrior is not None:
                order.append((2, warrior, warrior.warrior_data.speed))
        order.sort(key=lambda x: x[2], reverse=True)
        return [(player, warrior) for player, warrior, speed in order]
    
    def check_dep(self):
        dep1 = 0
        dep2 = 0
        for i in range(len(self.p1_disposition)):
            warrior = self.p1_disposition[i]
            if warrior is not None and warrior.health <= 0:
                dep1 += 1
                self.p1_disposition[i] = None
        for i in range(len(self.p2_disposition)):
            warrior = self.p2_disposition[i]

            if warrior is not None and warrior.health <= 0:
                dep2 += 1
                self.p2_disposition[i] = None
        return dep1,dep2
    
    def _check_end_conditions(self):
        if self.p1_deaths == MAX_DEATHS_PER_TEAM and self.p2_deaths == MAX_DEATHS_PER_TEAM:
            self.stats.empates += 1
            self.stats.partidas_por_muerte += 1
            self.ended = True
            self.winner = "Draw"
        elif self.p1_deaths == MAX_DEATHS_PER_TEAM:
            self.stats.p2_victories += 1
            self.stats.partidas_por_muerte += 1
            self.ended = True
            self.winner = "P2"
        elif self.p2_deaths == MAX_DEATHS_PER_TEAM:
            self.stats.p1_victories += 1
            self.stats.partidas_por_muerte += 1
            self.ended = True
            self.winner = "P1"
        elif self.turn_number > MAX_TURNS:
            self.stats.empates += 1
            self.stats.partidas_por_limite_turnos += 1
            self.ended = True
            self.winner = "Draw"

        if self.ended:
            self.stats.partidas += 1
            self.stats.p1_total_deaths += self.p1_deaths
            self.stats.p2_total_deaths += self.p2_deaths
            self.stats.total_turns += self.turn_number


    def _reward(self, **components):
        weighted = sum(REWARD_WEIGHTS[name] * value for name, value in components.items())
        return weighted - TURN_PENALTY


    def calculate_rewards(self, damage_p1, damage_p2, damage_avoided_p1
                          , damage_avoided_p2,healed_p1,healed_p2,shapping1,shapping2):
        newDeaths_p1, newDeaths_p2 = self.check_dep()
        self.p1_deaths += newDeaths_p1
        self.p2_deaths += newDeaths_p2
        self._check_end_conditions()
        win_p1 = WIN_REWARD if self.winner == "P1" else (-WIN_REWARD if self.winner == "P2" else 0)
        win_p2 = -win_p1
        rewardP1 = self._reward(damage=damage_p1 - damage_p2,deaths=newDeaths_p1 - newDeaths_p2,
                                win=win_p1,blocks=damage_avoided_p1,heal=healed_p1,
                                shaping_weight=DISCOUNT_FACTOR_TURN * shapping2 - shapping1)
        rewardP2 = self._reward(damage=damage_p2 - damage_p1,deaths=newDeaths_p2 - newDeaths_p1,
                                win=win_p2,blocks=damage_avoided_p2,heal=healed_p2,
                                shaping_weight=-(DISCOUNT_FACTOR_TURN * shapping2 - shapping1))
        return rewardP1, rewardP2