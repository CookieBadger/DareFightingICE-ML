from pyftg import AIInterface
from pyftg.struct import *
from pyftg import State
import numpy as np
import random
import time
import datetime
import os.path
from math import *
import io
import threading

import tkinter as tk
from PIL import Image, ImageTk
from tkinter import PhotoImage

## Improvement: 
# use energy levels and special action in states
# use stageX and stageY parameters
# print who won at end of round and end of game
# use center of hitbox instead of player.x and enemy.x

TRAINING : bool = True

# Training parameters
learning_rate = 0.1

# Environment parameters
gamma = 0.95

# Exploration parameters
max_epsilon = 1.0
min_epsilon = 0.05 
decay_rate = 0.01

## tutorial: https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial

class QLearningAI5(AIInterface):
    def __init__(self):
        super().__init__()
        self.blind_flag = False
        self.current_action = None
        self.last_action = None
        self.last_state = None
        self.qtable = None
        now = datetime.datetime.now()
        self.time_str = str(now.hour) + "-" + str(now.minute)
        self.last_reward_log_time = 0
        self.wins = 0
        self.losses = 0
        self.games_won = 0
        self.games_lost = 0
        self.episode = 0
        self.episode_filename = self.__class__.__name__ + "-episode"
        self.special_action_count = 0
        self.written = False
        self.forced_action = False
        self.force_count = 0

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
        player = self.frame_data.get_character(True) # todo, make dynamic to know which player we actually are
        enemy = self.frame_data.get_character(False)
        
        state = self.get_state(player, enemy)

        if TRAINING and self.current_action:
            if not self.forced_action: # only learn when action was not forced
                time_passed = self.frame_data.current_frame_number / 60
                max_hp = self.game_data.max_hps[1]
                enemy_health_lost =  (max_hp - enemy.hp)/max_hp
                player_health_lost =  (max_hp - player.hp)/max_hp
                enemy_health_lost_over_time = enemy_health_lost / time_passed
                player_health_lost_over_time = player_health_lost / time_passed
                energy_reward = 0.375*(min(player.energy,200)/200)
                reward = enemy_health_lost - player_health_lost + energy_reward
                self.learn(self.current_action, self.last_state, reward)
            self.current_action = None
        
        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
        else:
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*self.episode)
            action = self.force_action_policy(state)
            if action != None:
                self.forced_action = True
                self.force_count += 1
            else:
                action = self.epsilon_greedy_policy(state, epsilon)
                self.forced_action = False
            
            if action != self.last_action:
                self.key.empty()
                self.cc.skill_cancel()
                self.cc.command_call(action)
            self.current_action = action
            self.last_action = action

            # if last action, measure reward of last action
            # update Qtable with reward
            # choose action, remember state and action 
        self.last_state = state
        if not self.forced_action:
            self.force_count = 0
    
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
        self.current_action = None
        self.last_action = None
        if TRAINING:
            self.episode += 1
            self.log_episode()
        health_diff = round_result.remaining_hps[0] - round_result.remaining_hps[1]
        self.log(self.__class__.__name__ + "_health-diff-log_" + self.time_str, health_diff)

    def game_end(self):
        self.qtable.save()
        QTable.instance_nr = 0
        if self.wins > self.losses: 
            self.games_won+=1
        elif self.losses > self.wins:
            self.games_lost += 1
        
        print("---------------------------------")
        print("Game ended: ", self.__class__.__name__, "wins" if self.wins > self.losses else ("looses." if self.wins < self.losses else "ties with opponent."),\
               " (", self.wins, ":", self.losses, ")")
        print("Special Action Count:", self.special_action_count)
        print("Total games won vs lost: ", self.games_won, ":", self.games_lost)
        print("=================================")
        print(".")
        self.wins = 0
        self.losses = 0

    def force_action_policy(self, state):
        started_crouching = self.last_action == "CROUCH"
        started_idling = self.last_action == "5"
        if started_crouching and state[2] != State.CROUCH.value: # if we crouch, commit to it
            return "CROUCH" 
        if started_idling and state[2] == State.CROUCH.value:
            return "5" #idle
        if self.force_count < 1 and self.last_action == "FORWARD_WALK":
            return "FORWARD_WALK"
        return None

    def epsilon_greedy_policy(self, state, epsilon):
        random_f = random.uniform(0,1)
        if not TRAINING or random_f > epsilon:  
            action = self.qtable.get_best_action(state)
        else:
            action = self.random_action(state)
        if action == SPECIAL_ACTION[0]: 
            self.special_action_count += 1
        return action

    def greedy_policy(self, state):
        action = self.qtable.get_best_action(state)
        return action

    def random_action(self, state):
        energy_level = state[4]
        possible_actions = ["5"] # idle
        
        if state[2] == State.AIR.value:
            possible_actions = possible_actions + AIR_FREE_ACTIONS
            if energy_level >= 1:
                possible_actions = possible_actions + AIR_ENERGY_ACTIONS
        elif state[2] == State.STAND.value:
            possible_actions = possible_actions + STAND_FREE_ACTIONS
            if energy_level >= 1:
                possible_actions = possible_actions + STAND_ENERGY_ACTIONS
            if energy_level >= 2:
                possible_actions = possible_actions + SPECIAL_ACTION
        elif state[2] == State.CROUCH.value:
            possible_actions = possible_actions + CROUCH_ACTIONS + ["CROUCH"]

        return possible_actions[random.randint(0, len(possible_actions)-1)]

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
            self.log(self.__class__.__name__ + "_reward-log_" + self.time_str, new_reward) ## log reward
    
    def log(self, file_name, item):
        text = "{:.2f}, ".format(item)
        path = "logs/" + file_name
        if os.path.exists(path):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        f = open(path, append_write)
        f.write(text)
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


# Additional: IDLE ("5") possible in all states
STAND_ACTIONS = ["FORWARD_WALK","DASH","BACK_STEP","CROUCH","FOR_JUMP","BACK_JUMP","STAND_GUARD","THROW_A","THROW_B","STAND_A","STAND_B","STAND_FA","STAND_FB","STAND_D_DF_FA","STAND_D_DF_FB","STAND_F_D_DFA","STAND_F_D_DFB","STAND_D_DB_BA","STAND_D_DB_BB","STAND_D_DF_FC"]
STAND_FREE_ACTIONS=["FORWARD_WALK","DASH","BACK_STEP","CROUCH","FOR_JUMP","BACK_JUMP","STAND_GUARD","STAND_A","STAND_B","STAND_FA","STAND_FB","STAND_F_D_DFA"]
STAND_ENERGY_ACTIONS=["THROW_A","THROW_B","STAND_D_DF_FA","STAND_D_DF_FB","STAND_F_D_DFB","STAND_D_DB_BA","STAND_D_DB_BB"] # cost between 5 and 55 energy
SPECIAL_ACTION=["STAND_D_DF_FC"] # costs 150 energy

AIR_ACTIONS = ["AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "AIR_GUARD", "AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB"]
AIR_FREE_ACTIONS=["AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "AIR_GUARD"]
AIR_ENERGY_ACTIONS=["AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB"] # require between 5 and 50 energy

CROUCH_ACTIONS = ["CROUCH_GUARD","CROUCH_A","CROUCH_B","CROUCH_FA","CROUCH_FB"] # all free; also add CROUCH and IDLE

def get_all_actions():
    return STAND_ACTIONS + AIR_ACTIONS + CROUCH_ACTIONS + ["5"]

PLAYER_ENERGY = 3
ENEMY_ENERGY = 3

class QTable:
    instance_nr = 0
    def __init__(self):
        self.write = QTable.instance_nr % 2 == 0
        print("Qtable instance ", str(QTable.instance_nr), ", write: ", str(self.write))
        QTable.instance_nr = QTable.instance_nr + 1
        self.file_path = 'q_table_5.npy'
        try:
            # Load the Q-table from the file
            self.table = np.load(self.file_path)
        except FileNotFoundError:
            # doesn't exist
            #self.stand_table = np.zeros((len(STAND_ACTIONS)+1, PLAYER_ENEMY_X, PLAYER_ENEMY_Y, PLAYER_STATE, ENEMY_STATE, PLAYER_ENERGY, ENEMY_ENERGY))
            #self.air_table = np.zeros((len(AIR_ACTIONS)+1, PLAYER_ENEMY_X, PLAYER_ENEMY_Y, PLAYER_STATE, ENEMY_STATE, PLAYER_ENERGY, ENEMY_ENERGY))
            #self.crouch_table = np.zeros((len(CROUCH_ACTIONS)+2, PLAYER_ENEMY_X, PLAYER_ENEMY_Y, PLAYER_STATE, ENEMY_STATE, PLAYER_ENERGY, ENEMY_ENERGY))
            self.table = np.zeros((len(STAND_ACTIONS)+len(AIR_ACTIONS)+len(CROUCH_ACTIONS)+1, PLAYER_ENEMY_X, PLAYER_ENEMY_Y, PLAYER_STATE, ENEMY_STATE, PLAYER_ENERGY, ENEMY_ENERGY))
            
    
    def get_reward(self, action, state):
        action_idx = self.get_action_idx(action)
        reward = self.table[action_idx, state[0], state[1], state[2], state[3], state[4], state[5]]
        #print("Reward at [",action_idx,",",state[0],",",state[1],",",state[2],"]: ", reward)
        return reward
    
    def update(self, action, state, reward):
        action_idx = self.get_action_idx(action)
        self.table[action_idx, state[0], state[1], state[2], state[3], state[4], state[5]] = reward
    
    def get_best_action(self, state):
        return self.get_best(state)[0]
    
    def get_best_reward(self, state):
        return self.get_best(state)[1]
    
    def get_best(self, state):
        max_reward = 0.0
        max_action = "6" # right
        for action in get_all_actions():
            reward = self.get_reward(action, state)
            if reward > max_reward:
                max_reward = reward
                max_action = action
        return (max_action, max_reward)
    
    def get_action_idx(self, action):
        action_idx = 0
        if action in STAND_ACTIONS:
            action_idx = STAND_ACTIONS.index(action)
        elif action in AIR_ACTIONS:
            action_idx = len(STAND_ACTIONS)+AIR_ACTIONS.index(action)
        elif action in CROUCH_ACTIONS:
            action_idx = len(STAND_ACTIONS)+len(AIR_ACTIONS)+CROUCH_ACTIONS.index(action)
        else:
            action_idx = len(STAND_ACTIONS)+len(AIR_ACTIONS)+len(CROUCH_ACTIONS) # IDLE
        return action_idx
    
    def save(self):
        if(self.write):
            print("Updated table saved to ", self.file_path)
            np.save(self.file_path, self.table)
            
            # Load the Q-table from the file
            loaded_q_table = np.load(self.file_path)

            # Verify that the loaded Q-table is the same as the original one
            if np.array_equal(self.table, loaded_q_table):
                print("Savefile verified")