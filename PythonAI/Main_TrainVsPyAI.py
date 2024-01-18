import logging
import argparse
from pyftg import Gateway
from KickAI import KickAI
from IdleAI import IdleAI
from QLearningAI5 import QLearningAI5

def start_game(port: int):
    gateway = Gateway(port=port)
    character = 'ZEN'
    game_num = 10
    agent1 = QLearningAI5()
    agent2 = KickAI()
    ai_name1 = agent1.__class__.__name__
    ai_name2 = agent2.__class__.__name__
    gateway.register_ai(ai_name1, agent1)
    gateway.register_ai(ai_name2, agent2)
    gateway.run_game([character, character], [ai_name1, ai_name2], game_num)
    gateway.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--port', default=50051, type=int, help='Port used by DareFightingICE')
    args = parser.parse_args()
    logging.basicConfig(level=args.log)
    start_game(args.port)
