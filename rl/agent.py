import numpy as np
import utils
import random
import math


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.points = 0
        self.s = None
        self.a = None

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def dis(self,state):
        
        snake_head_x,snake_head_y, snake_body, food_x, food_y= state
         
        if snake_head_x == 40:
            adjoining_wall_x = 1
        elif snake_head_x == 480:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0
        if snake_head_y == 40:
            adjoining_wall_y = 1
        elif snake_head_y == 480:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0


        if food_x == snake_head_x:
            food_dir_x = 0
        elif food_x > snake_head_x:
            food_dir_x = 2
        else:
            food_dir_x = 1
        if food_y == snake_head_y:
            food_dir_y = 0
        elif food_y > snake_head_y:
            food_dir_y = 2
        else:
            food_dir_y = 1
        adjoining_body_right = 0
        adjoining_body_left = 0
        adjoining_body_bottom = 0
        adjoining_body_top = 0
        for (x, y) in snake_body:
            if snake_head_x + 40 == x and snake_head_y == y:
                adjoining_body_right = 1
            if snake_head_x - 40 == x and snake_head_y == y:
                adjoining_body_left = 1
            if snake_head_x == x and snake_head_y + 40 == y:
                adjoining_body_bottom = 1
            if snake_head_x == x and snake_head_y - 40 == y:
                adjoining_body_top = 1
            
        state = (adjoining_wall_x,adjoining_wall_y,food_dir_x,food_dir_y,adjoining_body_top,adjoining_body_bottom,adjoining_body_left,adjoining_body_right)
        return state




    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''


        current_state = self.dis(state)
        #. If the game is over, that is when the dead variable becomes true, you only need to update your Q table and reset the game.

        if dead:
            previous_state = self.dis(self.s)
            # getting the maximum value we can get in this state by visiting all neighboring cells (up down left right)
            move_up_Q_score = self.Q[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][0]
            move_down_Q_score = self.Q[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][1] 
            move_left_Q_score = self.Q[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][2] 
            move_right_Q_score = self.Q[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][3]  
            max_Q_score = max(move_up_Q_score,move_down_Q_score,move_left_Q_score,move_right_Q_score)
            reward = -1
            N_S_A = self.N[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a]
            alpha = (self.C)/(self.C + N_S_A)
            # update q table
            self.Q[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a] = self.Q[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a] + alpha*(reward + (self.gamma*max_Q_score) - self.Q[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a])
            self.reset()
            return
        #During training, your agent needs to update your Q-table first (this step is skipped when the initial state and action are None),

        if self.s != None and self.a != None and self._train:
            previous_state = self.dis(self.s)
            move_up_Q_score = self.Q[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][0]
            move_down_Q_score = self.Q[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][1] 
            move_left_Q_score = self.Q[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][2] 
            move_right_Q_score = self.Q[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][3]  
            max_Q_score = max(move_up_Q_score,move_down_Q_score,move_left_Q_score,move_right_Q_score)
            # rewards
            if points > self.points:
                reward = 1
            elif dead:
                reward = -1
            else:
                reward = -0.1
            N_S_A = self.N[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a]

            alpha = (self.C)/(self.C + self.N[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a])
            self.Q[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a] = self.Q[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a] + alpha*(reward + (self.gamma*max_Q_score) - self.Q[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a])



        score = []
        for i in range(4):
            N_val = self.N[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][i]
            Q_val = self.Q[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][i]
            if N_val < self.Ne and self._train:
                score.append(1) 
            else:
                score.append(Q_val) 

        max_score = max(score)
        for i in range(3, -1, -1):
            if score[i] == max_score:
                action = i
                break
        if self._train:
            self.N[current_state[0]][current_state[1]][current_state[2]][current_state[3]][current_state[4]][current_state[5]][current_state[6]][current_state[7]][action] += 1
        self.s = state
        self.a = action
        self.points = points

        return action
