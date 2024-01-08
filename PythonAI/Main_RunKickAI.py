import logging
import argparse
from KickAI import KickAI
from pyftg import Gateway

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--port', default=50051, type=int, help='Port used by DareFightingICE')
    #parser.add_argument('--a1', default=None, type=str, help='The AI name to use for player 1')
    #parser.add_argument('--a2', default=None, type=str, help='The AI name to use for player 2')
    args = parser.parse_args()
    logging.basicConfig(level=args.log)
    
    gateway = Gateway(port=args.port)
    character = 'ZEN'
    game_num = 1
    agent1 = KickAI()
    gateway.register_ai("KickAI", agent1)
    gateway.load_agent(["KickAI"])
    #gateway.load_agent([args.a1])
    gateway.start_ai()
    gateway.close()
