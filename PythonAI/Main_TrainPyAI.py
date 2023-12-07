import logging
import argparse
from QLearningAI import QLearningAI
from pyftg import Gateway

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--port', default=50051, type=int, help='Port used by DareFightingICE')
    parser.add_argument('--a1', default=None, type=str, help='The AI name to use for player 1')
    parser.add_argument('--a2', default=None, type=str, help='The AI name to use for player 2')
    args = parser.parse_args()
    logging.basicConfig(level=args.log)
    
    
    gateway = Gateway(port=args.port)
    character = 'ZEN'
    game_num = 3
    agent1 = QLearningAI()
    agent2 = QLearningAI()
    gateway.register_ai("QLearningAI", agent1)
    gateway.register_ai("QLearningAI (2)", agent2)
    gateway.run_game([character, character], ["QLearningAI", "QLearningAI (2)"], game_num)
    gateway.close()
