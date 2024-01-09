import logging
import argparse
from pyftg import Gateway
from KickAI import KickAI
from IdleAI import IdleAI
from DeepQLearningAI import DeepQLearningAI

def start_game(port: int):
    gateway = Gateway(port=port)
    character = 'ZEN'
    game_num = 1
    agent1 = DeepQLearningAI()
    agent2 = IdleAI()
    ai_name = agent2.__class__.__name__
    gateway.register_ai("QLearningAI5", agent1)
    gateway.register_ai(ai_name, agent2)
    gateway.run_game([character, character], ["QLearningAI5", ai_name], game_num)
    gateway.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--port', default=50052, type=int, help='Port used by DareFightingICE')
    args = parser.parse_args()
    logging.basicConfig(level=args.log)
    start_game(args.port)
