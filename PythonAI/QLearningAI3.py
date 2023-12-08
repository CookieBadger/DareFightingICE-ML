from pyftg import AIInterface
from pyftg.struct import *
from pyftg import State
import numpy as np
import random
import time
import datetime
import os.path
from math import *

## Improvement: 
# use energy levels and special action in states
# use stageX and stageY parameters
# print who won at end of round and end of game
# use center of hitbox instead of player.x and enemy.x

# Training parameters
learning_rate = 0.1

# Evaluation parameters
n_eval_episodes = 1000

# Environment parameters
gamma = 0.95
eval_seed = []

# Exploration parameters
max_epsilon = 1.0
min_epsilon = 0.05 
decay_rate = 0.005 



## tutorial: https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial

class QLearningAI3(AIInterface):
    def __init__(self):
        super().__init__()
        self.blind_flag = False
        self.last_action = None
        self.last_state = None
        self.qtable = None
        now = datetime.datetime.now()
        self.time_str = str(now.hour) + "-" + str(now.minute)
        self.last_reward_log_time = 0
        self.wins = 0
        self.losses = 0
        self.episode = 0
        self.episode_filename = self.__class__.__name__ + "-episode"

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return self.blind_flag

    def initialize(self, game_data: GameData, player_number: int):
        self.cc = CommandCenter()
        self.key = Key()
        self.player = player_number
        self.game_data = game_data
        self.qtable = QTable()
        self.load_episode()

    def input(self) -> Key:
        return self.key

    def get_information(self, frame_data: FrameData, is_control: bool, non_delay_frame_data: FrameData):
        self.frame_data = frame_data
        self.cc.set_frame_data(self.frame_data, self.player)

    def get_screen_data(self, screen_data: ScreenData):
        self.screen_data = screen_data

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data

    def processing(self):
        if self.frame_data.empty_flag or self.frame_data.current_frame_number <= 0:
            return
        
        # retrieve state
        player = self.frame_data.get_character(True)
        enemy = self.frame_data.get_character(False)
        
        state = self.get_state(player, enemy)

        if self.last_action:
            max_hp = self.game_data.max_hps[1]
            reward =  (max_hp - enemy.hp)/max_hp
            self.learn(self.last_action, self.last_state, reward)
            self.last_action = None
        
        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
        else:
            self.key.empty()
            self.cc.skill_cancel()
            
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*self.episode)
            action = self.epsilon_greedy_policy(state, epsilon)

            self.last_action = action

            self.cc.command_call(action)
            # if last action, measure reward of last action
            # update Qtable with reward
            # choose action, remember state and action 
        self.last_state = state
    
    def round_end(self, round_result: RoundResult):
        print("Round End. Finished Episode ", self.episode)
        print(round_result.remaining_hps[0])
        print(round_result.remaining_hps[1])
        print(round_result.elapsed_frame)
        if round_result.remaining_hps[0] > round_result.remaining_hps[1]:
            print(self.__class__.__name__, " won!")
            self.wins += 1
        elif round_result.remaining_hps[0] < round_result.remaining_hps[1]:
            print(self.__class__.__name__, " lost.")
            self.losses += 1
        else:
            print("Tie.")
        self.last_action = None
        self.episode += 1
        self.log_episode()

    def game_end(self):
        self.qtable.save()
        QTable.instance_nr = 0
        print("---------------------------------")
        print("Game ended: ", self.__class__.__name__, "wins" if self.wins > self.losses else ("looses." if self.wins < self.losses else "ties with opponent."),\
               " (", self.wins, ":", self.losses, ")")
        print("=================================")
        print(".")
        self.wins = 0
        self.losses = 0

    def epsilon_greedy_policy(self, state, epsilon):
        random_f = random.uniform(0,1)
        if random_f > epsilon:
            action = self.qtable.get_best_action(state)
        else:
            action = self.random_action()
        return action

    def greedy_policy(self, state):
        action = self.qtable.get_best_action(state)
        return action

    def random_action(self):
        return ACTIONS[random.randint(0, len(ACTIONS)-1)]

    def get_state(self, player, enemy):
        player_x = (player.right+player.left)/2
        player_y = (player.top+player.bottom)/2
        enemy_x = (enemy.right+enemy.left)/2
        enemy_y = (enemy.top+enemy.bottom)/2
        #player.hp, enemy.hp, player.energy, enemy.energy, self.last_action, player.state, player.action, enemy.state, enemy.action
        #print("pX: ", player_x, ", eX: ", enemy_x, ", pY: ", player_y, ", eY: ", enemy_y)
        player_enemy_y = enemy_y - player_y
        player_enemy_x = enemy_x - player_x
        #half_qx = (PLAYER_ENEMY_X-1)/2 # half the length of the playerEnemyX state
        #base=1.5
        #power = pow(base, half_qx)-1
        
        #quantized_player_enemy_x = round(min(max(player_enemy_x / GAME_WIDTH, -1), 1) * half_qx+half_qx)
        #h = (player_enemy_x / GAME_WIDTH * power)
        #sign = copysign(1, h)
        #player_enemy_x_log = sign * log(abs(h+sign), base)
        #quantized_player_enemy_x_log = round(player_enemy_x_log + half_qx)
        quantized_player_enemy_x_log = logarithmic_quantize_symmetric_around_zero(player_enemy_x, PLAYER_ENEMY_X, 1.5, GAME_WIDTH)
        
        half_qy = (PLAYER_ENEMY_Y-1)/2
        quantized_player_enemy_y = round(min(max(player_enemy_y / PLAY_HEIGHT, -1), 1) * half_qy+half_qy)

        player_energy = get_energy_level(player.energy)
        enemy_energy = get_energy_level(enemy.energy)

        return (quantized_player_enemy_x_log, quantized_player_enemy_y, player.state.value, enemy.state.value, player_energy, enemy_energy)

    def learn(self, action, state, reward):
        prev_reward = self.qtable.get_reward(action, state)
        new_reward = prev_reward + learning_rate * (reward + gamma * self.qtable.get_best_reward(state) - prev_reward)
        self.qtable.update(action, state, new_reward)
            
        current_time = time.time()
        if current_time - self.last_reward_log_time > 1:
            self.last_reward_log_time = current_time
            self.log_reward(new_reward)

    def log_reward(self, reward):
        #print("logged reward: {:.2f}".format(reward))
        file_name = "reward-log-" + self.time_str
        f = open(file_name, "a")
        f.write("{:.2f}, ".format(reward))
        f.close()
    
    def log_episode(self):
        #print("logged reward: {:.2f}".format(reward))
        f = open(self.episode_filename, "w")
        f.write(str(self.episode))
        f.close()
    
    def load_episode(self):
        if os.path.isfile(self.episode_filename):
            f = open(self.episode_filename, "r")
            e_str = f.read()
            if e_str:
                e = int(e_str)
                self.episode = e
                print("Continuing at episode ", e_str)
            f.close()


        

def logarithmic_quantize_symmetric_around_zero(x, quantization_steps, log_base, range):
    half_qx = (quantization_steps-1)/2 # half the length of the playerEnemyX state
    power = pow(log_base, half_qx)-1
    
    h = (x / range * power)
    sign = copysign(1, h)
    log_x = sign * log(abs(h+sign), log_base)
    return round(log_x + half_qx)

def get_energy_level(energy):
    if energy >= 150: return 2
    if energy >= 55: return 1
    return 0
    

# Multi-dimensional table for Q-learning. 
# Dimensions: 
# player position & enemy position
# OR: player-enemy vector? (probably better)
# action
# energy ()
# health (very roughly quantized)

# QTable dimensions (quantizations):
GAME_WIDTH = 960
PLAY_HEIGHT = 500

PLAYER_ENEMY_X = 21
PLAYER_ENEMY_Y = 5 # needs to be odd
PLAYER_STATE = len(State) # 4
ENEMY_STATE = len(State)


STAND_ACTIONS = ("FORWARD_WALK","DASH","BACK_STEP","CROUCH","FOR_JUMP","BACK_JUMP","STAND_GUARD","THROW_A","THROW_B","STAND_A","STAND_B","STAND_FA","STAND_FB","STAND_D_DF_FA","STAND_D_DF_FB","STAND_F_D_DFA","STAND_F_D_DFB","STAND_D_DB_BA","STAND_D_DB_BB","STAND_D_DF_FC")
STAND_FREE_ACTIONS=("FORWARD_WALK","DASH","BACK_STEP","CROUCH","FOR_JUMP","BACK_JUMP","STAND_GUARD","STAND_A","STAND_B","STAND_FA","STAND_FB","STAND_F_D_DFA")
STAND_ENERGY_ACTIONS=("THROW_A","THROW_B","STAND_D_DF_FA","STAND_D_DF_FB","STAND_F_D_DFB","STAND_D_DB_BA","STAND_D_DB_BB") # cost between 5 and 55 energy
SPECIAL_ACTION=("STAND_D_DF_FC") # costs 150 energy

AIR_ACTIONS = ("AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "AIR_GUARD", "AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB")
AIR_FREE_ACTIONS=("AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "AIR_GUARD")
AIR_ENERGY_ACTIONS=("AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB") # require between 5 and 50 energy

CROUCH_ACTIONS = ("CROUCH_GUARD","CROUCH_A","CROUCH_B","CROUCH_FA","CROUCH_FB") # all free


PLAYER_ENERGY = 3
ENEMY_ENERGY = 3

class QTable:
    instance_nr = 0
    def __init__(self):
        self.write = QTable.instance_nr % 2 == 0
        print("Qtable instance ", str(QTable.instance_nr), ", write: ", str(self.write))
        QTable.instance_nr = QTable.instance_nr + 1
        self.file_path = 'q_table_2.npy'
        try:
            # Load the Q-table from the file
            self.table = np.load(self.file_path)
        except FileNotFoundError:
            # doesn't exist
            self.table = np.zeros((len(ACTIONS), PLAYER_ENEMY_X, PLAYER_ENEMY_Y, PLAYER_STATE, ENEMY_STATE, PLAYER_ENERGY, ENEMY_ENERGY))
            
    
    def get_reward(self, action, state):
        action_idx = ACTIONS.index(action)
        reward = self.table[action_idx, state[0], state[1], state[2], state[3], state[4], state[5]]
        #print("Reward at [",action_idx,",",state[0],",",state[1],",",state[2],"]: ", reward)
        return reward
    
    def update(self, action, state, reward):
        action_idx = ACTIONS.index(action)
        self.table[action_idx, state[0], state[1], state[2], state[3], state[4], state[5]] = reward
    
    def get_best_action(self, state):
        return self.get_best(state)[0]
    
    def get_best_reward(self, state):
        return self.get_best(state)[1]
        
    def get_best(self, state):
        max_reward = 0.0
        max_action = "6" # right
        for action in ACTIONS:
            reward = self.get_reward(action, state)
            if reward > max_reward:
                max_reward = reward
                max_action = action
        return (max_action, max_reward)
    
    def save(self):
        if(self.write):
            print("Updated table saved to ", self.file_path)
            np.save(self.file_path, self.table)
            
            # Load the Q-table from the file
            loaded_q_table = np.load(self.file_path)

            # Verify that the loaded Q-table is the same as the original one
            if np.array_equal(self.table, loaded_q_table):
                print("Savefile verified")