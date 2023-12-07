from pyftg import AIInterface
from pyftg.struct import *
from pyftg import State
import numpy as np
import random
import time
import datetime
from math import *

# Training parameters
learning_rate = 0.7        

# Evaluation parameters
n_eval_episodes = 100      

# Environment parameters
gamma = 0.95
eval_seed = []

# Exploration parameters
max_epsilon = 1.0           
min_epsilon = 0.05           
decay_rate = 0.0005 
epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*0)



## tutorial: https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial

class QLearningAI(AIInterface):
    def __init__(self):
        super().__init__()
        self.blind_flag = False
        self.last_action = None
        self.last_state = None
        self.qtable = None
        now = datetime.datetime.now()
        self.time_str = str(now.hour) + "-" + str(now.minute)
        self.last_reward_log_time = 0   

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
        
        state = self.get_state(player.x, player.y, enemy.x, enemy.y, player.hp, enemy.hp, player.energy, enemy.energy, self.last_action, player.state, player.action, enemy.state, enemy.action)

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
            
            action = self.epsilon_greedy_policy(state, epsilon)

            self.last_action = action

            self.cc.command_call(action)
            # if last action, measure reward of last action
            # update Qtable with reward
            # choose action, remember state and action 
        self.last_state = state
        
    def choose_action(self):
        # choose best action or 
        # choose random action
        # depening on Exploration Exploitation, and whether state was discovered before
        pass
    
    def round_end(self, round_result: RoundResult):
        print(round_result.remaining_hps[0])
        print(round_result.remaining_hps[1])
        print(round_result.elapsed_frame)
        self.last_action = None

    def game_end(self):
        self.qtable.save()
        QTable.instance_nr = 0
        print("game ended")

    def epsilon_greedy_policy(self, state, epsilon):
        random_int = random.uniform(0,1)
        if random_int > epsilon:
            action = self.qtable.get_best_action(state)
        else:
            action = self.random_action()
        return action

    def greedy_policy(self, state):
        action = self.qtable.get_best_action(state)
        return action

    def random_action(self):
        return ACTIONS[random.randint(0, len(ACTIONS)-1)]

    def get_state(self, player_x, player_y, enemy_x, enemy_y, player_health, enemy_health, player_energy, enemy_energy, previous_action, player_state, player_action, enemy_state, enemy_action):
        #print("pX: ", player_x, ", eX: ", enemy_x, ", pY: ", player_y, ", eY: ", enemy_y)
        player_enemy_y = enemy_y - player_y
        player_enemy_x = enemy_x - player_x
        # quantized_player_x = round(min(max(player_x / GAME_WIDTH, 0), 1) * (PLAYER_X-1))
        # quantized_enemy_x = round(min(max(enemy_x / GAME_WIDTH, 0), 1) * (ENEMY_X-1))
        
        half_qx = (PLAYER_ENEMY_X-1)/2 # half the length of the playerEnemyX state
        base=1.5
        power = pow(base, half_qx)-1
        
        quantized_player_enemy_x = round(min(max(player_enemy_x / GAME_WIDTH, -1), 1) * half_qx+half_qx)
        h = (player_enemy_x / GAME_WIDTH * power)
        sign = copysign(1, h)
        player_enemy_x_log = sign * log(abs(h+sign), base)
        quantized_player_enemy_x_log = round(player_enemy_x_log + half_qx)
        quantized_player_enemy_x_log = logarithmic_quantize_symmetric_around_zero(player_enemy_x, PLAYER_ENEMY_X, 1.5, GAME_WIDTH)
        
        half_qy = (PLAYER_ENEMY_Y-1)/2
        quantized_player_enemy_y = round(min(max(player_enemy_y / PLAY_HEIGHT, -1), 1) * half_qy+half_qy)
        return (quantized_player_enemy_x_log, quantized_player_enemy_y, player_state.value, enemy_state.value)

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
        with open(file_name, "a") as myfile:
            myfile.write("{:.2f}, ".format(reward))
        

def logarithmic_quantize_symmetric_around_zero(x, quantization_steps, log_base, range):
    half_qx = (quantization_steps-1)/2 # half the length of the playerEnemyX state
    power = pow(log_base, half_qx)-1
    
    h = (x / range * power)
    sign = copysign(1, h)
    log_x = sign * log(abs(h+sign), log_base)
    return round(log_x + half_qx)
    

# Multi-dimensional table for Q-learning. 
# Dimensions: 
# player position & enemy position
# OR: player-enemy vector? (probably better)
# action
# energy ()
# health (very roughly quantized)

# QTable dimensions (quantizations):
GAME_WIDTH = 800
PLAY_HEIGHT = 200

PLAYER_ENEMY_X = 21
PLAYER_ENEMY_Y = 5 # needs to be odd
PLAYER_STATE = len(State) # 4
ENEMY_STATE = len(State)
ACTIONS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B") # ignore special action for now

class QTable:
    instance_nr = 0
    def __init__(self):
        self.write = QTable.instance_nr % 2 == 0
        print("Qtable instance ", str(QTable.instance_nr), ", write: ", str(self.write))
        QTable.instance_nr = QTable.instance_nr + 1
        self.file_path = 'q_table_1.npy'
        try:
            # Load the Q-table from the file
            self.table = np.load(self.file_path)
        except FileNotFoundError:
            # doesn't exist
            self.table = np.zeros((len(ACTIONS), PLAYER_ENEMY_X, PLAYER_ENEMY_Y, PLAYER_STATE, ENEMY_STATE))
            
    
    def get_reward(self, action, state):
        action_idx = ACTIONS.index(action)
        reward = self.table[action_idx, state[0], state[1], state[2], state[3]]
        #print("Reward at [",action_idx,",",state[0],",",state[1],",",state[2],"]: ", reward)
        return reward
    
    def update(self, action, state, reward):
        action_idx = ACTIONS.index(action)
        self.table[action_idx, state[0], state[1], state[2], state[3]] = reward
    
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