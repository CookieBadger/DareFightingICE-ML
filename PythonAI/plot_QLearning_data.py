import matplotlib.pyplot as plt
import os.path

reward_file_name = "logs/QLearningAI4_reward-log_12-55"

## Print rewards

plt.figure()
plt.tight_layout()

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
    plt.subplot(221)
    plt.plot(reward_array)
    plt.xlabel("seconds")
    plt.ylabel("reward")
    plt.title("Reward progression")
    #plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

health_diff_file_name = "logs/QLearningAI4_health-diff-log_12-55"
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
    plt.subplot(222)
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
    plt.subplot(223)
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
    print(n_av_array)
    print(len(av_array)-(n-1))
    plt.plot(range(len(av_array)-4), n_av_array, "r-")

plt.show()