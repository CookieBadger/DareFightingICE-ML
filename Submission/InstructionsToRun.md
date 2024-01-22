# Instructions to Run:
## Prerequisites
1. Have [Java](https://www.oracle.com/java/technologies/downloads/) (recommended version: 21) and [Python](https://www.python.org/downloads/) 3.11+ installed
2. Install the following python packages with `pip`:
    * `pyftg` (includes the interface for the environment and python)
    * `torch`
    * `numpy`
    * `pillow`
3. Have a working audio device connected (can be at volume 0, but if the game can't play audio, startup fails)

## Run the Environment
1. Download and unzip the file [DareFightingICE-v6.1.zip](https://github.com/TeamFightingICE/FightingICE/releases/tag/v6.1)
5. Run the `run-<your-os>-amd64` file inside the unzipped folder

## Run the AI
1. Open a terminal and navigate to the `AI` folder of this submission
2. Run the respective AI
    * If you want to run the Q-Learning AI, run `python ./Main_RunQLearningAI.py`
    * If you want to run the Deep Q-Learning AI, run `python ./Main_RunDeepQLearningAI.py`

## Start the Game
1. In the environment, navigate with the arrows and the key 'Z' to confirm. Select and confirm "Fight". 
2. The AI agent should be selected automatically. Select the MCTS agent in the respective field with the arrow keys. Leave the character selection default (ZEN).
3. Select "Play" and press 'Z'