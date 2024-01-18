from pyftg import AIInterface
from pyftg.struct import *
from pyftg import State

import random
import numpy as np
import math

import time
import datetime
import os.path

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image, ImageTk

from collections import namedtuple, deque
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINING = False


GAME_WIDTH = 960
PLAY_HEIGHT = 500

SCREENDATA_WIDTH = 32
SCREENDATA_HEIGHT = 32

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# MAX_EPSILON is the starting value of epsilon
# MIN_EPSILON is the final value of epsilon
# DECAY_RATE controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LEARNING_RATE is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
MAX_EPSILON = 0.9
MIN_EPSILON = 0.05
DECAY_RATE = 0.005
TAU = 0.005
LEARNING_RATE = 1e-4

# Additional: IDLE ("5") possible in all states
STAND_ACTIONS = ["FORWARD_WALK","DASH","BACK_STEP","CROUCH","FOR_JUMP","BACK_JUMP","STAND_GUARD","THROW_A","THROW_B","STAND_A","STAND_B","STAND_FA","STAND_FB","STAND_D_DF_FA","STAND_D_DF_FB","STAND_F_D_DFA","STAND_F_D_DFB","STAND_D_DB_BA","STAND_D_DB_BB","STAND_D_DF_FC"]
STAND_FREE_ACTIONS=["FORWARD_WALK","DASH","BACK_STEP","CROUCH","FOR_JUMP","BACK_JUMP","STAND_GUARD","STAND_A","STAND_B","STAND_FA","STAND_FB","STAND_F_D_DFA"]
STAND_ENERGY_ACTIONS=["THROW_A","THROW_B","STAND_D_DF_FA","STAND_D_DF_FB","STAND_F_D_DFB","STAND_D_DB_BA","STAND_D_DB_BB"] # cost between 5 and 55 energy
SPECIAL_ACTION=["STAND_D_DF_FC"] # costs 150 energy

AIR_ACTIONS = ["AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "AIR_GUARD", "AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB"]
AIR_FREE_ACTIONS=["AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "AIR_GUARD"]
AIR_ENERGY_ACTIONS=["AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB"] # require between 5 and 50 energy

CROUCH_ACTIONS = ["CROUCH_GUARD","CROUCH_A","CROUCH_B","CROUCH_FA","CROUCH_FB"] # all free; also add CROUCH and IDLE


i = 0
actions = {0: "5"}
stand_free_actions_nrs = []
stand_energy_actions_nrs = []
special_action_nrs = []
air_free_actions_nrs = []
air_energy_actions_nrs = []
crouch_actions_nrs = []
crouch_nr = 0

for value in STAND_FREE_ACTIONS:
    i+=1
    stand_free_actions_nrs += [i]
    actions[i] = value
    if value == "CROUCH":
        crouch_nr = i

for value in STAND_ENERGY_ACTIONS:
    i+=1
    stand_energy_actions_nrs += [i]
    actions[i] = value

for value in SPECIAL_ACTION:
    i+=1
    special_action_nrs += [i]
    actions[i] = value

for value in AIR_FREE_ACTIONS:
    i+=1
    air_free_actions_nrs += [i]
    actions[i] = value

for value in AIR_ENERGY_ACTIONS:
    i+=1
    air_energy_actions_nrs += [i]
    actions[i] = value

for value in CROUCH_ACTIONS:
    i+=1
    crouch_actions_nrs += [i]
    actions[i] = value

n_actions = len(actions)


class DeepQLearningAISingleImage(AIInterface):
    def __init__(self):
        super().__init__()
        self.blind_flag = False
        self.current_action_tensor = None
        self.last_action = None
        self.last_state = None
        self.forced_action = False
        self.force_count = 0
        self.model = None
        now = datetime.datetime.now()
        self.time_str = str(now.hour) + "-" + str(now.minute)
        self.last_reward_log_time = time.time()
        self.wins = 0
        self.losses = 0
        self.games_won = 0
        self.games_lost = 0
        self.episode = 0
        self.episode_filename = self.__class__.__name__ + "-episode"
        self.special_action_count = 0
        self.model_save_file = self.__class__.__name__ + ".pt"
        self.last_reward = 0

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return self.blind_flag

    def initialize(self, game_data: GameData, player_number: int):
        self.cc = CommandCenter()
        self.key = Key()
        self.player = player_number
        self.game_data = game_data
        self.load_episode()

        self.policy_net = DQN(n_actions, SCREENDATA_WIDTH, SCREENDATA_HEIGHT).to(device)
        if os.path.exists(self.model_save_file):
            self.policy_net.load_state_dict(torch.load(self.model_save_file))

        self.target_net = DQN(n_actions, SCREENDATA_WIDTH, SCREENDATA_HEIGHT).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def input(self) -> Key:
        return self.key

    def get_information(self, frame_data: FrameData, is_control: bool, non_delay_frame_data: FrameData):
        self.frame_data = frame_data
        self.cc.set_frame_data(self.frame_data, self.player)

    def get_screen_data(self, screen_data: ScreenData):
            self.screen_data = screen_data

    def get_audio_data(self, audio_data: AudioData):
            self.audio_data = audio_data

    def get_state(self):
        img_data = np.frombuffer(self.screen_data.display_bytes, dtype=np.uint8)
        img_data = img_data.reshape((32, 64, 3)) # 32x64 pixels, RGB (3 channels)

        # Create an image from the numpy array
        image = Image.fromarray(img_data, 'RGB')
        image_array = np.array(image)
        image_array = image_array[0:32, 32:64]

        image_array = image_array / 255 # convert to 0..1
        image_tensor = torch.from_numpy(image_array).float()

        mean = [0.50032169, 0.5016008, 0.50118719]
        std = [0.07209322, 0.07115186, 0.06815479]
        image_tensor = (image_tensor - torch.tensor(mean)) / torch.tensor(std)

        # move channels to end and add batch dimension
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor 


    def processing(self):
        if self.frame_data.empty_flag or self.frame_data.current_frame_number <= 0 or not self.screen_data:
            return
        
        player_state = self.frame_data.get_character(True) # todo, make dynamic to know which player we actually are

        if TRAINING and self.current_action_tensor != None and self.last_state != None and self.screen_data != None:
            if not self.forced_action:
                reward = self.get_reward()
                self.last_reward = reward
                reward_tensor = torch.tensor([reward], device=device, dtype=torch.long)
                next_state = self.get_state()
                self.memory.push(self.last_state, self.current_action_tensor, next_state, reward_tensor)
                
                self.optimize_model()
                
                # Soft-update target network
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                self.log_reward(reward)
            
            self.current_action_tensor = None

        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
        else:
            
            epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-self.episode * DECAY_RATE)
            
            action = self.force_action_policy(player_state)
            if action != None:
                self.forced_action = True
                self.force_count += 1
            else:
                action_tensor = self.epsilon_greedy_policy(self.get_state(), epsilon, player_state)
                action = actions[action_tensor.item()]
                self.forced_action = False
            
                self.current_action_tensor = action_tensor
                self.last_state = self.get_state()
            
            if action != self.last_action:
                self.key.empty()
                self.cc.skill_cancel()
                self.cc.command_call(action)

            self.last_action = action
        
        if not self.forced_action:
            self.force_count = 0
    
    def force_action_policy(self, state):
        started_crouching = self.last_action == "CROUCH"
        started_idling = self.last_action == "5"
        stance = state.state.value
        if started_crouching and stance != State.CROUCH.value: # if we crouch, commit to it
            return "CROUCH" 
        if started_idling and stance == State.CROUCH.value:
            return "5" #idle
        if self.force_count < 1 and self.last_action == "FORWARD_WALK":
            return "FORWARD_WALK"
        return None
    
    def epsilon_greedy_policy(self, state, epsilon, player_state):
        random_f = random.uniform(0,1)
        if random_f > epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                temp = self.policy_net(state)
                action =  temp.max(1).indices.view(1, 1)
        else:
            action = self.random_action(player_state)
        return action
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
             return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        next_states = torch.cat([s for s in batch.next_state])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        #print("reward batch: ", reward_batch.size())
        # Huber loss
        criterion = nn.SmoothL1Loss()
        #print("calculating loss: nextstate_values ", next_state_values.size(), "; expected_state_action: ", expected_state_action_values.size())
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) ### check (debug) dimensions. C:\Users\Emil\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\loss.py:933: UserWarning: Using a target size (torch.Size([128, 1, 128])) that is different to the input size (torch.Size([128, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size. return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def random_action(self, state):
        possible_actions = [0] # idle
        stance = state.state.value
        energy = state.energy

        if stance == State.AIR.value:
            possible_actions = possible_actions + air_free_actions_nrs
            if energy >= 55:
                possible_actions = possible_actions + air_energy_actions_nrs
        elif stance == State.STAND.value:
            possible_actions = possible_actions +  stand_free_actions_nrs
            if energy >= 55:
                possible_actions = possible_actions + stand_energy_actions_nrs
            if energy >= 150:
                possible_actions = possible_actions + special_action_nrs
        elif stance == State.CROUCH.value:
            possible_actions = possible_actions + crouch_actions_nrs + [crouch_nr]

        action_nr = possible_actions[random.randint(0, len(possible_actions)-1)]
    
        return torch.tensor([[action_nr]], device=device, dtype=torch.long)

    def get_reward(self):
        player = self.frame_data.get_character(True)
        enemy = self.frame_data.get_character(False)
        max_hp = self.game_data.max_hps[1]
        enemy_health_lost =  (max_hp - enemy.hp)/max_hp
        player_health_lost =  (max_hp - player.hp)/max_hp
        #energy_reward = 0.375*(min(player.energy,200)/200)
        return enemy_health_lost - player_health_lost #+ energy_reward
        
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
        self.current_action_tensor = None
        self.last_action = None
        if TRAINING:
            self.episode += 1
            self.log_episode()
        health_diff = round_result.remaining_hps[0] - round_result.remaining_hps[1]
        self.log(self.__class__.__name__ + "_health-diff-log_" + self.time_str, health_diff)
    
    
    def game_end(self):
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
        if TRAINING:
            torch.save(self.policy_net.state_dict(), self.model_save_file)  
    
    def log_reward(self, new_reward):
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
        self.log(self.__class__.__name__ + "_episode-reward-log_" + self.time_str, self.last_reward)
    
    def load_episode(self):
        if os.path.isfile(self.episode_filename):
            f = open(self.episode_filename, "r")
            e_str = f.read()
            if e_str:
                e = int(e_str)
                self.episode = e
                print("Continuing at episode ", e_str)
            f.close()


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.memory = deque([], maxlen = capacity)
	
	def push (self, *args):
		# Save a transition
		self.memory.append(Transition(*args))

	def sample (self, batch_size):
		return random.sample(self.memory, batch_size)
	
	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):

    def __init__(self, actions, obsv_width, obsv_height):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pre_linear_width = round(obsv_width/(2**3))
        self.pre_linear_height = round(obsv_height/(2**3))
        self.fc1 = nn.Linear(128 * self.pre_linear_width * self.pre_linear_height, 512)  # Adjust the input features based on the output of the last pooling layer
        self.fc2 = nn.Linear(512, actions) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * self.pre_linear_width * self.pre_linear_height)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x





""" episode = 0
episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf()) """
