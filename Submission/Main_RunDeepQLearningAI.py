import logging
import argparse
from DeepQLearningAI import DeepQLearningAI
from pyftg import Gateway

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--port', default=50051, type=int, help='Port used by DareFightingICE')
    args = parser.parse_args()
    logging.basicConfig(level=args.log)
    
    gateway = Gateway(port=args.port)
    character = 'ZEN'
    game_num = 1
    agent1 = DeepQLearningAI()
    ai_name1 = agent1.__class__.__name__
    gateway.register_ai(ai_name1, agent1)
    gateway.load_agent([ai_name1])
    gateway.start_ai()
    gateway.close()
