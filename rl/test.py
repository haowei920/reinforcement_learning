import argparse
import mp7
Args = argparse.Namespace(show_eps = 10,human = False, test_eps=1000, train_eps=1000, window=1000,C=40, Ne=40, food_x=80, food_y=80, gamma=0.7, model_name='q_agent.npy', snake_head_x=200, snake_head_y=200)
App = mp7.Application(Args)
App.execute()
