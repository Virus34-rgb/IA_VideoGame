import os
import random
import time

from Antiguos.choose_state import Choose_state
from Antiguos.observation import Observation
from Antiguos.playerAI import PlayerAI
from constants import EPISODES_RANGE_POOL, EPISODES_SAVE_MODEL, POOL_PORCENTAGE


class Trainer:
    def __init__(self, player1, player2, environment,opponent_pool, train_episodes, eval_episodes,
                 pathp1_1, pathp1_2, pathp2_1, pathp2_2, path_stats, path_stats2,
                 logger=None, snapshot_every=1000, progress_every=50):
        self.player1 = player1
        self.player2 = player2
        self.environment = environment
        self.opponent_pool = opponent_pool
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.pathp1_1 = pathp1_1
        self.pathp1_2 = pathp1_2
        self.pathp2_1 = pathp2_1
        self.pathp2_2 = pathp2_2
        self.path_stats = path_stats
        self.path_stats2 = path_stats2
        self.logger = logger
        self.snapshot_every = snapshot_every
        self.progress_every = progress_every

    def train(self):
        self._load_if_exists()
        self._run(
            episodes=self.train_episodes,
            epsilon_turn=0.5,
            epsilon_sel=None,
            learn_p1=True,
            learn_p2=True,
            stats_path=self.path_stats,
            restore_epsilon=True,
        )
        self._save_if_supported(self.player1, self.pathp1_1, self.pathp1_2)
        self._save_if_supported(self.player2, self.pathp2_1, self.pathp2_2)

    def evaluate(self):
        self.environment.stats.reset()
        self._load_if_exists()
        self._run(
            episodes=self.eval_episodes,
            epsilon_turn=0.02,
            epsilon_sel=0.02,
            learn_p1=False,
            learn_p2=False,
            stats_path=self.path_stats2,
            restore_epsilon=True,
        )
        # no se guarda modelo: evaluate() no entrena nada

    def run_asymmetric(self, episodes, epsilon_turn, epsilon_sel, learn_p1, learn_p2, stats_path):
        """
        Entrada genérica para steps donde P1 y P2 no aprenden simétricamente
        (p.ej. entrenar contra un jugador humano, o solo actualizar P2).
        No carga ni guarda modelos: eso lo decide quien llame, ya que puede
        que P1/P2 de este step no sean self.player1/self.player2 "oficiales".
        """
        self._run(
            episodes=episodes,
            epsilon_turn=epsilon_turn,
            epsilon_sel=epsilon_sel,
            learn_p1=learn_p1,
            learn_p2=learn_p2,
            stats_path=stats_path,
            restore_epsilon=True,
        )

    def _load_if_exists(self):
        if hasattr(self.player1, "load_model") and os.path.exists(self.pathp1_1) and os.path.exists(self.pathp1_2):
            self.player1.load_model(self.pathp1_1, self.pathp1_2)
        if hasattr(self.player2, "load_model") and os.path.exists(self.pathp2_1) and os.path.exists(self.pathp2_2):
            self.player2.load_model(self.pathp2_1, self.pathp2_2)

    @staticmethod
    def _save_if_supported(player, path1, path2):
        if hasattr(player, "save_model"):
            player.save_model(path1, path2)

    def _run(self, episodes, epsilon_turn, epsilon_sel, learn_p1, learn_p2, stats_path, restore_epsilon):
        backup = self._set_epsilons(epsilon_turn, epsilon_sel)
        start_time = time.time()
        p2_training_player = self.player2
        opponent_from_pool = False
        for episode in range(episodes):
            if(episode != 0 and episode % EPISODES_SAVE_MODEL  == 0):
                self.opponent_pool.save_version(p2_training_player) #Del backup porque p2 podria ser la copia de la que se esta entrenando
            if(episode != 0 and episode % (EPISODES_RANGE_POOL) == 0):
                opponent_from_pool = False
                self.player2 = p2_training_player
                if(random.random() <= POOL_PORCENTAGE):
                    cantidad,_,_ = self.opponent_pool.list_models()
                    if(cantidad > 0):
                        opponent_from_pool = True
                        p2_training_player = self.player2
                        path_sel, path_turn = self.opponent_pool.get_random()
                        self.player2 = PlayerAI()
                        self.player2.load_model(path_sel,path_turn)
            self._run_episode(episode, learn_p1, learn_p2,opponent_from_pool)
            if self.logger and (learn_p1 or learn_p2) and self.snapshot_every and episode % self.snapshot_every == 0:
                self.logger.log_snapshot(episode, self.player1, p2_training_player, self.environment.stats)
            self._print_progress(episode, episodes, start_time)

        if episodes > 0:
            print()  # salto de línea tras la barra de progreso
        self.environment.stats.guardar_stats(stats_path, self.environment.warriors_classes)
        if restore_epsilon:
            self._restore_epsilons(backup)

    def _set_epsilons(self, epsilon_turn, epsilon_sel):
        backup = {}
        for name, player in (("p1", self.player1), ("p2", self.player2)):
            if hasattr(player, "epsilon_turn"):
                backup[f"{name}_turn"] = player.epsilon_turn
                player.epsilon_turn = epsilon_turn
            if epsilon_sel is not None and hasattr(player, "epsilon_sel"):
                backup[f"{name}_sel"] = player.epsilon_sel
                player.epsilon_sel = epsilon_sel
        return backup

    def _restore_epsilons(self, backup):
        for name, player in (("p1", self.player1), ("p2", self.player2)):
            if f"{name}_turn" in backup:
                player.epsilon_turn = backup[f"{name}_turn"]
            if f"{name}_sel" in backup:
                player.epsilon_sel = backup[f"{name}_sel"]

    def _run_episode(self, episode, learn_p1, learn_p2,opponent_from_pool):
        state = self.environment.reset()
        cstates1, actions1, cstates2, actions2 = self._select_teams(state)
        observation1, observation2 = self.getObservation(state)
        reward1_acum = reward2_acum = 0

        while not self.environment.ended:
            action_p1 = self.player1.turn(observation1)
            action_p2 = self.player2.turn(observation2)
            state, reward1, reward2, ended = self.environment.turn(action_p1, action_p2)
            next_obs1, next_obs2 = self.getObservation(state)

            if learn_p1:
                self.player1.remember_turn(observation1, action_p1, reward1, next_obs1, ended)
            if learn_p2 and not opponent_from_pool:
                self.player2.remember_turn(observation2, action_p2, reward2, next_obs2, ended)

            observation1, observation2 = next_obs1, next_obs2
            reward1_acum += reward1
            reward2_acum += reward2

        if learn_p1:
            loss1_turn = self.player1.replay_turn()
            if self.logger:
                self.logger.log_loss(episode, self.player1.replayed_turn, "p1", "turn", loss1_turn)
            self._remember_and_replay_selection(cstates1, actions1, reward1_acum, self.player1, "p1", episode)
            if(self.train_episodes != 0):
                self.player1.update_beta()
                self.player1.update_epsilon()

        if learn_p2 and not opponent_from_pool:
            loss2_turn = self.player2.replay_turn()
            if self.logger:
                self.logger.log_loss(episode, self.player2.replayed_turn, "p2", "turn", loss2_turn)
            self._remember_and_replay_selection(cstates2, actions2, reward2_acum, self.player2, "p2", episode)
            if(self.train_episodes != 0):
                self.player2.update_beta()
                self.player2.update_epsilon()

        # guarda el total de reward de la partida completa, aprenda o no
        self.environment.stats.total_reward_p1 += reward1_acum
        self.environment.stats.total_reward_p2 += reward2_acum

    def _select_teams(self, state):
        cstate1, cstate2 = self.createChooseState(state, 0, 0, 0, 0)
        warr1_1, pos1_1, action1_1 = self.player1.selection(cstate1)
        warr2_1, pos2_1, action2_1 = self.player2.selection(cstate2)
        state = self.environment.team_selection(warr1_1, pos1_1, warr2_1, pos2_1)

        cstate1_2, cstate2_2 = self.createChooseState(state, warr2_1, pos2_1 + 1, warr1_1, pos1_1 + 1)
        warr1_2, pos1_2, action1_2 = self.player1.selection(cstate1_2)
        warr2_2, pos2_2, action2_2 = self.player2.selection(cstate2_2)
        state = self.environment.team_selection(warr1_2, pos1_2, warr2_2, pos2_2)

        # Se repiten los primeros seleccionados porque queremos que las IAs solo conozcan el primero
        cstate1_3, cstate2_3 = self.createChooseState(state, warr2_1, pos2_1 + 1, warr1_1, pos1_1 + 1)
        warr1_3, pos1_3, action1_3 = self.player1.selection(cstate1_3)
        warr2_3, pos2_3, action2_3 = self.player2.selection(cstate2_3)
        self.environment.team_selection(warr1_3, pos1_3, warr2_3, pos2_3)

        cstates1 = (cstate1, cstate1_2, cstate1_3)
        actions1 = (action1_1, action1_2, action1_3)
        cstates2 = (cstate2, cstate2_2, cstate2_3)
        actions2 = (action2_1, action2_2, action2_3)
        return cstates1, actions1, cstates2, actions2

    def _remember_and_replay_selection(self, cstates, actions, reward_acum, player, player_name, episode):
        c1, c2, c3 = cstates
        a1, a2, a3 = actions
        transitions = [(c1, a1, c2, False), (c2, a2, c3, False), (c3, a3, None, True)]
        for c, a, next_c, done in transitions:
            player.remember_selection(c, a, reward_acum, next_c, done)
            loss = player.replay_selection()
            if self.logger:
                self.logger.log_loss(episode, player.replayed_selection, player_name, "selection", loss)

    def createChooseState(self, state, p1_first_warrior, p1_first_pos, p2_first_warrior, p2_first_pos):
        p1_state = Choose_state(state.p1_disposition, self.environment.warriors_classes, p2_first_warrior, p2_first_pos)
        p2_state = Choose_state(state.p2_disposition, self.environment.warriors_classes, p1_first_warrior, p1_first_pos)
        return p1_state, p2_state

    def getObservation(self, state):
        p1_life, p1_distribution = self.opp_calculate(state.p1_disposition)
        p2_life, p2_distribution = self.opp_calculate(state.p2_disposition)
        observation1 = Observation(state.p1_disposition, p2_life, p2_distribution, self.environment.turn_number)
        observation2 = Observation(state.p2_disposition, p1_life, p1_distribution, self.environment.turn_number)
        return observation1, observation2

    def opp_calculate(self, disposition):
        dispositionId = []
        life = []
        for w in disposition:
            dispositionId.append(w.warrior_data.id if w is not None else None)
            life.append(w.health / w.warrior_data.max_health if w is not None else 0)
        return life, dispositionId

    def _print_progress(self, episode, total_episodes, start_time):
        if total_episodes == 0:
            return
        if self.progress_every and episode % self.progress_every != 0 and episode != total_episodes - 1:
            return
        elapsed = time.time() - start_time
        pct = (episode + 1) / total_episodes * 100
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        remaining = total_episodes - episode - 1
        eta = remaining / eps_per_sec if eps_per_sec > 0 else 0
        print(
            f"\r[{pct:5.1f}%] Episodio {episode + 1}/{total_episodes} "
            f"| {eps_per_sec:6.1f} ep/s | ETA {self._format_time(eta)}   ",
            end="", flush=True
        )

    @staticmethod
    def _format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.0f}s"
        if minutes > 0:
            return f"{minutes}m {secs:.0f}s"
        return f"{secs:.1f}s"