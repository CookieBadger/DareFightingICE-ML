import matplotlib.pyplot as plt
import os.path
import numpy as np

#reward_file_name = "logs/QLearningAI5_reward-log_11-47"
#reward_file_name = "logs/QLearningAI5_proc2evaluation_reward-log_14-0"
#reward_file_name = "logs/DeepQLearningAISimplified_reward-log_15-33"
#reward_file_name = "logs/DeepQLearningAISimplified2_reward-log_18-7"
#reward_file_name = "logs/DeepQLearningAI_reward-log_15-33"
reward_file_name = "logs/DeepQLearningAISingleImage_reward-log_0-11"

## Print rewards

plt.figure()
plt.tight_layout()

def step(x: float):
    if x > 0:
        return 1.0
    return 0.0

if os.path.isfile(reward_file_name):
    f = open(reward_file_name, "r")
    reward_array = []
    str = f.read()
    if str:
        for e in str.split(','):
            e=e.strip()
            if e:
                reward_array.append(float(e))
    f.close()
    plt.subplot(231)
    plt.plot(reward_array)
    plt.xlabel("seconds")
    plt.ylabel("reward")
    plt.title("Reward progression")
    #plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

ep_reward_file_name = "logs/DeepQLearningAISingleImage_episode-reward-log_0-11"
if os.path.isfile(ep_reward_file_name):
    f = open(ep_reward_file_name, "r")
    cum_array = []
    str = f.read()
    if str:
        for e in str.split(','):
            e=e.strip()
            if e:
                cum = cum_array[len(cum_array)-1] if len(cum_array)>0 else 0
                cum_array.append(cum + float(e))
    f.close()
    plt.subplot(236)
    plt.plot(cum_array)
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.title("Reward progression")

#health_diff_file_name = "logs/QLearningAI5_proc2evaluation_health-diff-log_14-0"
#health_diff_file_name = "logs/QLearningAI5_health-diff-log_12-52"
#health_diff_file_name = "logs/DeepQLearningAISimplified2_health-diff-log_18-7"
#health_diff_file_name = "logs/DeepQLearningAI_health-diff-log_15-33"
#health_diff_file_name = "logs/DeepQLearningAISimplified_health-diff-log_15-33"
health_diff_file_name = "logs/DeepQLearningAISingleImage_health-diff-log_0-11"
if os.path.isfile(health_diff_file_name):
    f = open(health_diff_file_name, "r")
    str = f.read()
    diff_array = []
    if str:
        for e in str.split(','):
            e=e.strip()
            if e:
                diff_array.append(float(e))
    f.close()
    plt.subplot(232)
    plt.bar(range(len(diff_array)), diff_array)
    plt.xlabel("round nr")
    plt.ylabel("player-enemy health difference")
    plt.title("Performance over Rounds")
    #plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    av_array = []
    i=2
    while i < len(diff_array) :
        avg = (diff_array[i-2]+ diff_array[i-1]+ diff_array[i])/3
        av_array.append(avg)
        i += 3
    f.close()
    plt.subplot(233)
    plt.bar(range(len(av_array)), av_array)
    plt.xlabel("game nr")
    plt.ylabel("player-enemy health difference")
    plt.title("Performance over Games (avg of 3 rounds)")
    #plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    n=5
    n_av_array = []
    i=n-1
    while i < len(av_array) :
        sum = 0
        for j in range(n):
            sum += av_array[i-(j)]
        avg = sum / n
        n_av_array.append(avg)
        i += 1
    plt.plot(range(len(av_array)-4), n_av_array, "r-")

    
    win_array = []
    i=2
    while i < len(diff_array) :
        wins = (np.sign(diff_array[i-2])+ np.sign(diff_array[i-1])+ np.sign(diff_array[i]))
        win_array.append(wins)
        i += 3
    f.close()
    plt.subplot(234)
    plt.bar(range(len(win_array)), win_array)
    plt.xlabel("game nr")
    plt.ylabel("Win-loss difference")
    plt.title("Performance over Games (rounds won minus rounds lost)")
    #plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    win_array = []
    n_games_per_bar = 5
    i=n_games_per_bar*3-1
    while i < len(diff_array) :
        d5 = 0
        for j in range(n_games_per_bar):
            d5 += (np.sign(diff_array[i-2-j*3])+ np.sign(diff_array[i-1-j*3])+ np.sign(diff_array[i-j*3]))
        win_array.append(d5)
        i += n_games_per_bar*3
    f.close()
    plt.subplot(235)
    plt.bar(range(len(win_array)), win_array)
    plt.xlabel("game nr")
    plt.ylabel("Win-loss difference")
    plt.title("Best of 5 (rounds won minus rounds lost in {:d} games)".format(n_games_per_bar))
    #plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


plt.show()