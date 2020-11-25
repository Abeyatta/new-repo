
# coding: utf-8

# In[1]:


from itertools import product
import random 
import re
import pandas as pd
import numpy as np
from math import factorial as fact
from matplotlib import pyplot as plt
from numpy.linalg import matrix_power

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 200)


# In[2]:


class Prob_Player_Wins_points:
    
    """ This class returns probability of a player reaching to every "point score" permutations on the way to winning the game.
     i.e 0,15,30,40 and win"""
    
    def __init__ (self, p1, p2):
        
        """ 
        The constructor for Prob_Player_Wins_points class. 
        Parameters: 
           float (p1): prob of player1 winning when serving. 
           float (p2): prob of player2 winning when serving.   

        """     
        self.p1 = p1       
        self.p2 = p2
        
    def permutation(self, a, b):
        """no repetition, order doesn't matter. a & b must be int
        Parameters: int(a, b)
        Returns: factorial(a) / (factorial(a-b)*factorial(b))"""
        
        self.a = a
        self.b = b
        return fact(a) / (fact(a - b) * fact(b))
    

    def P1_Game_Love(self):
        # win all of the first 4 serves
        return self.p1**4

    def P1_Game_15(self):
        # win the last point and any 3 from the first 4
        return self.p1**4 * (1 - self.p1) * self.permutation(4,3)

    def P1_Game_30(self):
        # win the last point and any 3 from the first 5
        return self.p1**4 * (1 - self.p1)**2 * self.permutation(5,3)

    def P1_To_Deuce(self):
        # each wins 3 of the first 6
        return self.p1**3 * (1 - self.p1)**3 * self.permutation(6,3)
    
    def P1_From_Deuce(self):
        """ win two consecetive points after getting to deuce 
        ie. 2 * ((win*lose) or (lose*win)) """
        return self.P1_To_Deuce() * self.p1**2 / (self.p1**2 + (1 - self.p1)**2) 

    def P1_Win_Game(self):
        """ add all the above winning probabilities"""
        return self.p1**4 * (15 - 24*self.p1 + 10*self.p1**2 + (20*self.p1*(1-self.p1)**3 / (self.p1**2 + (1-self.p1)**2)))    
    
    
    # Player2 win game with prob p2 when serves
    def P2_Game_Love(self):
        # win all 
        return self.p2**4

    def P2_Game_15(self):
        # win the last point and any 3 from the first 4
        return self.p2**4 * (1 - self.p2) * self.permutation(4,3)

    def P2_Game_30(self):
        # win the last point and any 3 from the first 5
        return self.p2**4 * (1 - self.p2)**2 * self.permutation(5,3)
    
    def P2_To_Deuce(self):
        # each wins 3 of the first 6
        return self.p2**3 * (1 - self.p2)**3 * self.permutation(6,3)
    
    def P2_From_Deuce(self):
        """ win two consecetive points after getting to deuce 
        ie. 2 * ((win+lose) or (lose+win)) """
        return self.P2_To_Deuce() * self.p2**2 / (self.p2**2 + (1 - self.p2)**2)     

    def P2_Win_Game(self):
        """ add all the above winning probabilities"""
        return self.p2**4 * (15 - 24*self.p2 + 10*self.p2**2 + (20*self.p2*(1-self.p2)**3 / (self.p2**2 + (1-self.p2)**2)))


# In[3]:


# Create a new Prob_Player_Wins_points object with p1, p2 given
PP = Prob_Player_Wins_points(0.60, 0.65)

# Call P1_From_Deuce() method
PP.P1_From_Deuce()
PP.P1_Win_Game()


# In[4]:


class Tran_Matrix_Point(Prob_Player_Wins_points):
    
    """ This class creates a state transition probability matrix of a Markov chain gives the probabilities of 
    transitioning from one state to another in a single time unit. 
    i.e player1 wins a point from a given initial point with p1 or lose with (1-p1)"""
    
    # half_state
    half_state = [0,15,30,40]
    # Cartesian product (every possible combination of scores) from half_state
    score = list(product(half_state,half_state))
    # scores as a dataframe,  like == 15:40
    df = pd.DataFrame(score,columns=("opp","own"))
    # array of all possible scores execpet the deuce/winning score (win + lose)
    state = (df['own'].map(str) + ':' + df['opp'].map(str)).tolist()
    # states: All states including win/lose
    states = state.copy()
    states.extend(['won', 'lost'])
    
    def matr_zero(self):
        
        """Return: Null Transition Matrix of dimension len(states)"""
        tm = pd.DataFrame(0, index=self.states, columns=self.states) 
        return tm
    
    def state_conc(self, own, opp):
        
        """ Return: point scored separated by ':' """
        self.own = own
        self.opp = opp
        return str(self.half_state[own]) + ':' + str(self.half_state[opp])
    
    def Prob_of_Win_point(self):
        """ Return: transition matrix where every row add up to 1"""
        tm = self.matr_zero()
        for s in self.states:
            if s in self.state:
                own = self.half_state.index(int(re.findall('[0-9]+', s)[0]))
                opp = self.half_state.index(int(re.findall('[0-9]+', s)[1]))

                # own ==> half_stateate = [0,1,2] ==> scores = [0,15,30]
                if own < 3:
                    # invoked state_concatenet fun
                    j_win = self.states.index(self.state_conc(own + 1, opp))
                else:
                    if opp < 3:
                        j_win = self.states.index('won')
                    else:
                        j_win = self.states.index('40:30')

                # opp ==> half_stateate = [0,1,2] ==> scores = [0,15,30]        
                if opp < 3:
                    j_lost = self.states.index(self.state_conc(own, opp + 1))
                else:
                    if own < 3:
                        j_lost = self.states.index('lost')
                    else:
                        j_lost = self.states.index('30:40')


                r = tm.columns.get_loc(s)
                tm.iloc[r, j_win] = self.p1
                tm.iloc[r, j_lost] = 1 - self.p1


            else:
                if s == 'won':
                    tm.iloc[tm.columns.get_loc('won'), tm.columns.get_loc('won')] = 1
                else:
                    tm.iloc[tm.columns.get_loc('lost'), tm.columns.get_loc('lost')] = 1

        return tm 

test = Tran_Matrix_Point(0.45, 0.65)
print(test.Prob_of_Win_point.__doc__)


# In[5]:


# win game from point (a,b) when Player A is serving
gg = Tran_Matrix_Point(0.60, 0.45)
M = gg.Prob_of_Win_point()
Final_prob = pd.DataFrame(matrix_power(M, 20), index=M.columns, columns=M.columns)#[['Win_A','Win_B']][:-2]
Final_prob[['won','lost']][:-2]


# In[6]:


class Prob_P1_Wins_TieBreak(Prob_Player_Wins_points):

    """ This class returns probability of a player reaching to every "point score" permutations on the way to 
    winning a tie break. i.e 1,2,3,4,5,6 and win"""    
    
    # In all tie breaks Player1 assumed to start the serving 
    def P1_TieBr7_Love(self):
        # Serve sequence: p1 p2p2 p1p1 p2p2 ==> All 7 points won by Player1
        # Player1 p1, Player2 p2 wins with p1 and p2 respectively 
        return self.p1**3 * (1 - self.p2)**4

    def P1_TieBr7_1(self):
       
        """ Win the last point and six of the previous 7
        From self.p1 self.p2self.p2 self.p1self.p1 self.p2self.p2 win all 3 of own serves and 3 out of 4 opponent's 
        serve AND win all 4 of opponent's serves and 2 out of 3 own serves
        self.p1 * self.permutation(3,3)*self.p1**3 * self.permutation(4,3)*(1-self.p2)**3 * self.p2 + 
        self.permutation(4,3) *
        self.permutation(4,4) * (1-self.p2)**4 * self.permutation(3,2) * self.p1**2 * (1-self.p1) """
        
        return (4 * self.p1**4 * (1 - self.p2)**3 * self.p2) + (3 * self.p1**3 *(1 - self.p2)**4 * (1 - self.p1))

    def P1_TieBr7_2(self):
        
        part_1 = (self.p1*(self.permutation(4,4) * self.p1**4 * (1 - self.p1)** 0 * 
                           self.permutation(4,2)*(1-self.p2)**2 * self.p2**2))
        
        part_2 = (self.p1*(self.permutation(4,3) * self.p1**3 * (1 - self.p1)**1 * 
                           self.permutation(4,3)*(1-self.p2)**3 * self.p2**1))
        
        part_3 = (self.p1*(self.permutation(4,2) * self.p1**2 * (1 - self.p1)**2 * 
                           self.permutation(4,4)*(1-self.p2)**4 * self.p2**0))
        return part_1 + part_2 + part_3
    
    def P1_TieBr7_3(self):
        part_1 = ((1-self.p1)*(self.permutation(5,5) * self.p1**5 * (1 - self.p1)**0 * 
                               self.permutation(4,1)*(1-self.p2)**1 * self.p2**3))
        part_2 = ((1-self.p1)*(self.permutation(5,4) * self.p1**4 * (1 - self.p1)**1 * 
                               self.permutation(4,2)*(1-self.p2)**2 * self.p2**2))
        part_3 = ((1-self.p1)*(self.permutation(5,3) * self.p1**3 * (1 - self.p1)**2 * 
                               self.permutation(4,3)*(1-self.p2)**3 * self.p2**1))
        part_4 = ((1-self.p1)*(self.permutation(5,2) * self.p1**2 * (1 - self.p1)**3 * 
                               self.permutation(4,4)*(1-self.p2)**4 * self.p2**0))
        return part_1 + part_2 + part_3 + part_4

    def P1_TieBr7_4(self):
        part_1 = ((1-self.p1)*(self.permutation(5,5) * self.p1**5 * (1 - self.p1)**0 * 
                               self.permutation(5,1)*(1-self.p2)**1 * self.p2**5))
        part_2 = ((1-self.p1)*(self.permutation(5,4) * self.p1**4 * (1 - self.p1)**1 * 
                               self.permutation(5,2)*(1-self.p2)**2 * self.p2**4))
        part_3 = ((1-self.p1)*(self.permutation(5,3) * self.p1**3 * (1 - self.p1)**2 * 
                               self.permutation(5,3)*(1-self.p2)**3 * self.p2**3))
        part_4 = ((1-self.p1)*(self.permutation(5,2) * self.p1**2 * (1 - self.p1)**3 * 
                               self.permutation(5,4)*(1-self.p2)**4 * self.p2**2))
        part_5 = ((1-self.p1)*(self.permutation(5,1) * self.p1**1 * (1 - self.p1)**4 * 
                               self.permutation(5,5)*(1-self.p2)**5 * self.p2**1))
        return part_1 + part_2 + part_3 + part_4 + part_5   
    
    def P1_TieBr7_5(self):
        part_1 = (self.p1*(self.permutation(5,5) * self.p1**5 * (1 - self.p1)**0 * 
                           self.permutation(6,1)*(1-self.p2)**1 * self.p2**5))
        part_2 = (self.p1*(self.permutation(5,4) * self.p1**4 * (1 - self.p1)**1 * 
                           self.permutation(6,2)*(1-self.p2)**2 * self.p2**4))
        part_3 = (self.p1*(self.permutation(5,3) * self.p1**3 * (1 - self.p1)**2 * 
                           self.permutation(6,3)*(1-self.p2)**3 * self.p2**3))
        part_4 = (self.p1*(self.permutation(5,2) * self.p1**2 * (1 - self.p1)**3 * 
                           self.permutation(6,4)*(1-self.p2)**4 * self.p2**2))
        part_5 = (self.p1*(self.permutation(5,1) * self.p1**1 * (1 - self.p1)**4 * 
                           self.permutation(6,5)*(1-self.p2)**5 * self.p2**1))
        part_6 = (self.p1*(self.permutation(5,0) * self.p1**0 * (1 - self.p1)**5 * 
                           self.permutation(6,6)*(1-self.p2)**6 * self.p2**0))  
        return part_1 + part_2 + part_3 + part_4 + part_5 + part_6 
        
    def P1_TieBr6_6(self):
        part_1 = (self.permutation(6,6) * self.p1**6 * (1 - self.p1)**0 * 
                  self.permutation(6,0)*(1-self.p2)**0 * self.p2**6)
        part_2 = (self.permutation(6,5) * self.p1**5 * (1 - self.p1)**1 * 
                  self.permutation(6,1)*(1-self.p2)**1 * self.p2**5)
        part_3 = (self.permutation(6,4) * self.p1**4 * (1 - self.p1)**2 * 
                  self.permutation(6,2)*(1-self.p2)**2 * self.p2**4)
        part_4 = (self.permutation(6,3) * self.p1**3 * (1 - self.p1)**3 * 
                  self.permutation(6,3)*(1-self.p2)**3 * self.p2**3)    
        part_5 = (self.permutation(6,2) * self.p1**2 * (1 - self.p1)**4 * 
                  self.permutation(6,4)*(1-self.p2)**4 * self.p2**2)
        part_6 = (self.permutation(6,1) * self.p1**1 * (1 - self.p1)**5 * 
                  self.permutation(6,5)*(1-self.p2)**5 * self.p2**1)
        part_7 = (self.permutation(6,0) * self.p1**0 * (1 - self.p1)**6 * 
                  self.permutation(6,6)*(1-self.p2)**6 * self.p2**0)
        return part_1 + part_2 + part_3 + part_4 + part_5 + part_6 + part_7 
    
    def P1_TieBr_from_deuce(self):
        return (self.p1 * (1- self.p2)) / (self.p1 + self.p2 - 2*self.p1*self.p2)
    
    def P1_TieBr7_6(self):
        return self.P1_TieBr_from_deuce() * self.P1_TieBr6_6()

    def P1_wins_Tiebreak(self):
        return (self.P1_TieBr7_Love() + self.P1_TieBr7_1() + self.P1_TieBr7_2() + 
                self.P1_TieBr7_3() + self.P1_TieBr7_4() + self.P1_TieBr7_5() + self.P1_TieBr7_6())
tb = Prob_P1_Wins_TieBreak(0.62, 0.60)
tb.P1_TieBr7_4()


# In[157]:


class Transition_Matrix_P1_Wins_TieBreak(Prob_Player_Wins_points):
    """ This class creates a transition matrix from wheich we can read 
    the probabilities of winning a tiebreaker game from any point score during a tie break. 
    Example; probability of winning the tie break from (5:4)...
    """
    # half_state
    half_state_tb = [0,1,2,3,4,5,6]
    sco = list(product(half_state_tb,half_state_tb))
    # This are all point scores when player A (Starts serving the tie break) serves. 
    A_serve = ['0:0', '0:3', '0:4', '1:2', '1:3', '1:6', '2:1', '2:2', '2:5', '2:6', '3:0',
           '3:1', '3:4', '3:5', '4:0', '4:3', '4:4', '5:2', '5:3', '5:6', '6:1', '6:2', '6:5', '6:6'] 
    # This are all point scores when player B (Not starts serving the tie break) serves.     
    B_serve = ['0:1', '0:2', '0:5', '0:6', '1:0', '1:1', '1:4', '1:5', '2:0', '2:3', '2:4', '3:2',
           '3:3', '3:6', '4:1', '4:2', '4:5', '4:6', '5:0', '5:1', '5:4', '5:5', '6:0', '6:3', '6:4']
    
    # scores as a dataframe,  like == 15:40
    df = pd.DataFrame(sco,columns=("opp","own"))
    
    # array of all possible scores execpet the deuce/winning score (win + lose)
    tb_state = (df['own'].map(str) + ':' + df['opp'].map(str)).tolist()
    
    # All tb_states including win/lose
    tb_states = tb_state.copy()
    tb_states.extend(['won', 'lost'])
    
    def matr_zero(self):
        # Null Transition Matrix
        tm = pd.DataFrame(0, index=self.tb_states, columns=self.tb_states)  

        return tm
    
    def state_conc(self, a, b):
        self.a = a
        self.b = b
        return str(self.half_state_tb[a]) + ':' + str(self.half_state_tb[b])
 

    def Tran_Matrix_Tb_prob(self):
        
        # create a 7 by 7 zero matrix 
        tm = self.matr_zero()

        for s in self.tb_states:
            
            if s in self.A_serve:
                
                A_own = self.half_state_tb.index(int(re.findall('[0-9]+', s)[0]))
                A_opp = self.half_state_tb.index(int(re.findall('[0-9]+', s)[1]))

                if A_own < 6:
                    A_win = self.tb_states.index(self.state_conc(A_own+1, A_opp))
                else:
                    if A_opp < 6:
                        A_win = self.tb_states.index('won')
                    else:
                        A_win = self.tb_states.index('6:5')

                if A_opp < 6:
                    A_lost = self.tb_states.index(self.state_conc(A_own, A_opp+1))
                else:
                    if A_own < 6:
                        A_lost = self.tb_states.index('lost')
                    else:
                        A_lost = self.tb_states.index('5:6')

                r = tm.columns.get_loc(s)
                tm.iloc[r, A_win] = self.p1
                tm.iloc[r, A_lost] = 1 - self.p1               

            elif s in self.B_serve:
                B_own = self.half_state_tb.index(int(re.findall('[0-9]+', s)[0]))
                B_opp = self.half_state_tb.index(int(re.findall('[0-9]+', s)[1])) 

                if B_own < 6:
                    A_win = self.tb_states.index(self.state_conc(B_own+1, B_opp))
                else:
                    A_win = self.tb_states.index('won')    
                    
                        
                if B_opp < 6:
                    A_lost = self.tb_states.index(self.state_conc(B_own, B_opp+1))
                else:
                    A_lost = self.tb_states.index('lost')                
                        


                r = tm.columns.get_loc(s)
                tm.iloc[r, A_win] = 1 - self.p2
                tm.iloc[r, A_lost] = self.p2

            else:
                if s == 'won':
                    tm.iloc[tm.columns.get_loc('won'), tm.columns.get_loc('won')] = 1
                else:
                    tm.iloc[tm.columns.get_loc('lost'), tm.columns.get_loc('lost')] = 1


        return tm 


# In[445]:


tran_mat = Transition_Matrix_P1_Wins_TieBreak(0.62, 0.60)
prob_tb = tran_mat.Tran_Matrix_Tb_prob()
result = pd.DataFrame(matrix_power(prob_tb, 20), index=prob_tb.columns, columns=prob_tb.columns)
#result.loc[:, ['won', 'lost']][:-2]


# In[ ]:


class Score_list:
    """ This class return all posible scores from which player A serving during tie break
    following a tie break rules where player A start the game by serving.
    i.e [A BB AA BB AA BB AA ...]
    player A serves when score at (a, b) and a + b is in [0, 3,4,7,8,11,12,15]"""
  
    def create_list(self, idx):
        lst = []
        self.idx = idx
        # number of states == 8
        for x in range(self.idx):
            for y in range(self.idx):
                if x + y < self.idx and x < 8 and y < 8:
                    lst.append(str(x) + ':' + str(y))
                else:
                    break
        return lst

    def create_score(self, idx):
        self.idx = idx
        mm = self.create_list(self.idx-2)
        m = [x for x in self.create_list(idx) if x not in mm]
        return m

l = Score_list()
print(l.create_list(3))
print('***************************')
print(l.create_score(0)+l.create_score(3)+l.create_score(4)+l.create_score(7)+l.create_score(8)+
      l.create_score(11)+l.create_score(12)+l.create_score(15))


# In[441]:


class Next_point_prob_tb(Prob_Player_Wins_points):
    """ This class creates a transition matrix from wheich we can read 
    the probability of reaching various score lines in a tiebreaker game. 
    """
    # half_state
    half_state_tb = [0,1,2,3,4,5,6,7]
    sco = list(product(half_state_tb,half_state_tb))
    # This are all point scores when player A (Starts serving the tie break) serves. 
    A_serve = ['0:0', '0:3', '0:4', '1:2', '1:3', '1:6', '2:1', '2:2', '2:5', '2:6', '3:0',
           '3:1', '3:4', '3:5', '4:0', '4:3', '4:4', '5:2',  '5:3', '5:6', '6:1', 
               '6:2', '6:5', '6:6', '4:7', '7:4','7:5', '5:7', '1:7','7:1','7:0', '0:7'] 
    
    # This are all point scores when player B (Not starts serving the tie break) serves.     
    B_serve = ['0:1', '0:2', '0:5', '0:6', '1:0', '1:1', '1:4', '1:5', '2:0', '2:3', '2:4', '3:2',
           '3:3', '3:6', '4:1', '4:2', '4:5', '4:6', '5:0', '5:1', '5:4', '5:5', '6:0', '6:3', '6:4',
              '2:7','7:2','7:3','3:7','7:6','6:7','7:7']

    
    # scores as a dataframe,  like == 15:40
    df = pd.DataFrame(sco,columns=("opp","own"))
    
    # array of all possible scores execpet the deuce/winning score (win + lose)
    tb_state = (df['own'].map(str) + ':' + df['opp'].map(str)).tolist()
    
    # All tb_states including win/lose
    tb_states = tb_state.copy()
    


    # All tb_states including win/lose
    ss = sorted(A_serve + B_serve)
    t = ['2:7','7:2','7:3','3:7','4:7', '7:4','7:5', '5:7', '1:7','7:1','7:0', '0:7']
    def matr_zero(self):
        # Null Transition Matrix
        tm = pd.DataFrame(0, index=self.tb_states, columns=self.tb_states)  

        return tm
    
    def state_conc(self, a, b):
        self.a = a
        self.b = b
        return str(self.half_state_tb[a]) + ':' + str(self.half_state_tb[b])
 

    def Tran_Matrix_Tb_prob(self):
        
        # create a 7 by 7 zero matrix 
        tm = self.matr_zero()

        for s in self.tb_states:
            
            if s in self.A_serve:
                
                A_own = half_state_tb.index(int(re.findall('[0-9]+', s)[0]))
                A_opp = half_state_tb.index(int(re.findall('[0-9]+', s)[1]))
        
                if A_own < 7:
                    A_win = self.tb_states.index(state_conc(A_own+1, A_opp))
                else:
                    A_win = self.tb_states.index(state_conc(A_own, A_opp))
                    
                if A_opp < 7:
                    A_loses = self.tb_states.index(state_conc(A_own, A_opp+1))
                else:
                    A_loses = self.tb_states.index(state_conc(A_own, A_opp))
                   
                r = tm.columns.get_loc(s)
                tm.iloc[r, A_win] = self.p1
                tm.iloc[r, A_loses] = 1 - self.p1

            else:
                B_own = self.half_state_tb.index(int(re.findall('[0-9]+', s)[0]))
                B_opp = self.half_state_tb.index(int(re.findall('[0-9]+', s)[1])) 

                if B_own < 7:
                    A_win = self.tb_states.index(state_conc(B_own+1, B_opp))
                else:
                    A_win = self.tb_states.index(state_conc(B_own, B_opp))
                
                if B_opp < 7:
                    A_loses = self.tb_states.index(state_conc(B_own, B_opp+1))
                else:
                    A_loses = self.tb_states.index(state_conc(B_own, B_opp))

         
                r = tm.columns.get_loc(s)
                tm.iloc[r, A_win] = 1 - self.p2
                tm.iloc[r, A_loses] = self.p2
                
        for i in self.t:
            for j in tm.columns:
                if i == j:
                    tm.loc[i, j] = 1
                else:
                    tm.loc[i, j] = 0

        return tm
    
    def next_score(self, p1_score,p2_score):
        self.p1_score = p1_score
        self.p2_score = p2_score
        score_len = self.p1_score + self.p2_score
        str_score = str(self.p1_score) + ':' + str(self.p2_score)
        m = self.Tran_Matrix_Tb_prob()
        col_score = m.columns
        vec = pd.DataFrame(matrix_power(self.Tran_Matrix_Tb_prob(), score_len), index=col_score, columns=col_score)
        prob_score = round(vec.iloc[:1][str_score][0], 4)
    
        return prob_score


# In[ ]:


# win game from point (a,b) when Player A is serving
b = Next_point_prob_tb(0.62, 0.60)
m = b.Tran_Matrix_Tb_prob()
b.next_score(3, 5)


# In[69]:


a_serve = l.create_score(1)+l.create_score(5)+l.create_score(9)+l.create_score(13)
b_serve = l.create_score(3)+l.create_score(7)+l.create_score(11)+l.create_score(15)


# In[ ]:


class Prob_P1_Wins_Set(Prob_P1_Wins_TieBreak):
    
    """ This class returns probability of a player winning a set"""    
    
    def P1_Set_6_0(self):
        # win all 6 games
        return self.P1_Win_Game()**6 
        
    def P1_Set_6_1(self):
        return (self.P1_Win_Game())**6 * (1 - self.P1_Win_Game()) * self.permutation(6,5)
    
    def P1_Set_6_2(self):
        return (self.P1_Win_Game() **6) * ((1 - self.P1_Win_Game()) ** 2) * self.permutation(7,5)
    
    def P1_Set_6_3(self):
        # win last game and any 5 of the first 8
        return (self.P1_Win_Game()**6) * ((1 - self.P1_Win_Game())**3) * self.permutation(8,5)
         
    def P1_Set_6_4(self):
        return (self.P1_Win_Game()**6) * ((1 - self.P1_Win_Game())**4) * self.permutation(9,5) 

    
    # SCORE 5:5 BOTH PLAYERS WIN 5 GAMES EACH
    def P1_Game5_5(self):
        return (self.P1_Win_Game()**5) * ((1 - self.P1_Win_Game())**5) * self.permutation(10,5)
    
    def P1_Set_7_5(self):
        return (self.P1_Win_Game())**2 * self.P1_Game5_5()  
        
    
    # SYNTHETIC GAMES. SETS WON AT 7:5, 8:6, 9:7, 10:8, ...
    def P1_synth_Game(self):
        exp1 = self.P1_Win_Game() * (1 - self.P2_Win_Game())
        exp2 =  (self.P1_Win_Game() + self.P2_Win_Game()) - 2*(self.P2_Win_Game() * self.P1_Win_Game())
        return self.P1_Game_greater_than5_O(6) * (exp1 / exp2)
        
    
    def P1_Game_greater_than5_5(self, m):
        # SCORE 6:6, 7:7, 8:8, ... BOTH PLAYERS WIN 5 GAMES EACH
        self.m = m
        assert type(m) == int, "Use integer values greater than 5"
        assert m >= 6
        return (2 * self.P1_Win_Game() * (1 - self.P1_Win_Game()))**(m - 5) * (self.P1_Game5_5()) 
    
    def P1_Win_Set_with_total_Game_greater_than11(self, n, m):
        # win set 8:6, 9:7, 10:8, ... 
        self.n = n
        self.m = m
        assert type(n) == int, "Use integer values for score!"
        assert type(m) == int, "Use integer values for score!"
        assert n == m + 2, "The difference must be two !"
        return (self.P1_Win_Game())**2 * self.P1_Game_greater_than5_5(m)

    
    def P1_Set_7_6(self, m):
        # Reach to 6:6 then win via tie break
        return self.P1_Game_greater_than5_5(6) *  self.P1_wins_Tiebreak()     
    
    def P1_win_set(self):
        
        return (self.P1_Set_6_0() + self.P1_Set_6_1() + self.P1_Set_6_2() + self.P1_Set_6_3() + 
                self.P1_Set_6_4() + self.P1_Set_7_5() + self.P1_Set_7_6(6))    
    


# In[ ]:


s = Prob_P1_Wins_Set(0.46, 0.90)
s.P1_win_set()


# In[ ]:


s = Prob_P1_Wins_Set(0.90, 0.46)
s.P1_win_set()


# In[ ]:


gg = Tennis_Prob_Games(0.65, 0.76)
print('win 7:0 ', gg.P1_TieBr7_Love())
print('win 7:1 ', gg.P1_TieBr7_1())
print('win 7:2 ', gg.P1_TieBr7_2())
print('win 7:3 ', gg.P1_TieBr7_3())
print('win 7:4 ', gg.P1_TieBr7_4())
print('win 7:5 ', gg.P1_TieBr7_5())
print('win 6:6 ', gg.P1_TieBr6_6())
print('win 7:6 ', gg.P1_TieBr7_6())


# In[ ]:


gg = Tennis_Prob_Games(0.65, 0.96)
gg.P1_win_set()


# In[ ]:


class Prob_P1_Wins_Match(Prob_P1_Wins_Set):
    
    """ This class returns probability of a player winning a match"""    
    def P1_Match_2_0(self):
        return self.P1_win_set()**2
        
    def P1_Match_2_1(self):
        return self.permutation(2,1) * (self.P1_win_set()**2) * (1 - self.P1_win_set())
    
    def P1_Match_3_0(self):
        return self.P1_win_set()**3
    
    def P1_Match_3_1(self):
        return self.permutation(3,2) * (self.P1_win_set()**3) * (1 - self.P1_win_set())
    
    def P1_Match_3_2(self):
        return self.permutation(4,2) * (self.P1_win_set()**3) * (1 - self.P1_win_set())**2   
    
    def P1_Men_win_match(self):
        # Men only
        return self.P1_Match_3_0() + self.P1_Match_3_1() + self.P1_Match_3_2()
    
    def P1_Women_win_match(self):
        # Women only
        return self.P1_Match_2_0() + self.P1_Match_2_1() 


# In[ ]:


gg = Tennis_Prob_match(0.60, 0.36)
gg.P1_Men_win_match()


# In[ ]:


lst = np.arange(start = 0.0, stop = 1, step = 0.01)
g = Tennis_Prob_Points(lst, 0.5)
s = Tennis_Prob_Games(lst, 0.5)
m = Tennis_Prob_match(lst, 0.50)

game_prob = g.P1_Win_Game()
set_prob = s.P1_win_set()
match_prob_W = m.P1_Women_win_match()
match_prob_M = m.P1_Men_win_match()

print(l.shape, game_prob.shape, set_prob.shape, match_prob_W.shape, match_prob_M.shape)

plt.rcParams["figure.figsize"] = (20,10)
plt.plot(l, game_prob, label='Game')
plt.plot(l, set_prob, label='Set')
plt.plot(l, match_prob_M, label='Match_M')
plt.plot(l, match_prob_W, label='Match_W')
plt.xlabel('Serve_Prob == p')
plt.ylabel('Win_Prob')
plt.title('Prob of winning Game, Set and Match for Expected values of p')
plt.legend()
plt.grid(True)


# In[ ]:


prob_player_win = pd.DataFrame({'Points':l.reshape(-1).tolist(),
             'Games':game_prob.reshape(-1).tolist(),
             'Sets':set_prob.reshape(-1).tolist(),
             'Match_W':match_prob_W.reshape(-1).tolist(),
             'Match_M':match_prob_M.reshape(-1).tolist()})
prob_player_win.iloc[40:61]


# In[ ]:


PP = Tennis_Prob_Points(0.60, 0.45)
print(PP.P1_Win_Game())
print(PP.P2_Win_Game())


# In[ ]:


PP = Tennis_Prob_Points(0.65, 0.75)
data = {'Game_6:6':[0, PP.P2_Win_Game(), (1 - PP.P2_Win_Game()), 0, 0],
       'Adv_A':[PP.P1_Win_Game(),0,0,0,0],
       'Adv_B':[(1-PP.P1_Win_Game()),0,0,0,0],
        'Win_A':[0, (1-PP.P2_Win_Game()),0,1,0],
        'Win_B':[0, 0, PP.P2_Win_Game(),0,1]}
ma = pd.DataFrame(data, index=['Game_6:6', 'Adv_A', 'Adv_B', 'Win_A', 'Win_B'])
ma


# In[ ]:


Final_prob = pd.DataFrame(matrix_power(ma, 20), index=ma.columns, columns=ma.columns)
Final_prob[['Win_A','Win_B']][:-2]


# In[ ]:


gg = Tennis_Prob_Games(0.62, 0.60)
print(gg.P1_Win_Game(), gg.P2_Win_Game())
print('win 7:0 ', gg.P1_TieBr7_Love())
print('win 7:1 ', gg.P1_TieBr7_1())
print('win 7:2 ', gg.P1_TieBr7_2())
print('win 7:3 ', gg.P1_TieBr7_3())
print('win 7:4 ', gg.P1_TieBr7_4())
print('win 7:5 ', gg.P1_TieBr7_5())
print('win 6:6 ', gg.P1_TieBr6_6())
print('win 7:6 ', gg.P1_TieBr7_6())


# In[ ]:


g = Tennis_Prob_Games_Counter(0.65, 0.65)
print('set 6:0 ', g.P1_Set_To_Love_Counter())
print('set 6:1 ', g.P1_Set_6_1_Counter())
print('set 6:2 ', g.P1_Set_6_2_Counter())
print(g.P1_Win_Game(), g.P2_Win_Game())


# In[ ]:


class Tennis_Prob_Games_Counter(Tennis_Prob_Points):
    
    def __init__ (self, p1, p2):
        self.p1 = p1       # prob of a player winning a serve
        self.p2 = p2
        
    def permutation(self, a, b):
        """no repetition, order doesn't matter. a & b must be int"""
        self.a = a
        self.b = b
        return fact(a) / (fact(a - b) * fact(b))
    
    
    
    # SCORE 5:5 BOTH PLAYERS WIN 5 GAMES EACH
    def P1_Game5_O(self):
        
        part_1 = self.P1_Win_Game()**5 * self.P2_Win_Game()**5
        
        part_2 = (self.permutation(5,4)**2 * self.P1_Win_Game()**4 * (1 - self.P1_Win_Game()) * 
                  self.P2_Win_Game()**4 * (1 - self.P2_Win_Game()) )
                  
        part_3 = (self.permutation(5,3)**2 * self.P1_Win_Game()**3 * (1 - self.P1_Win_Game())**2 * 
                  self.P2_Win_Game()**3 * (1 - self.P2_Win_Game())**2 )
                  
        part_4 = (self.permutation(5,2)**2 * self.P1_Win_Game()**2 * (1 - self.P1_Win_Game())**3 * 
                  self.P2_Win_Game()**2 * (1 - self.P2_Win_Game())**3 )
                  
        part_5 = (self.permutation(5,1)**2 * self.P1_Win_Game()**1 * (1 - self.P1_Win_Game())**4 * 
                  self.P2_Win_Game()**1 * (1 - self.P2_Win_Game())**4 )
                  
        part_6 = (1 - self.P1_Win_Game())**5 * (1 - self.P2_Win_Game())**5
                  
        return part_1 + part_2 + part_3 + part_4 + part_5 + part_6 
    
    
    def P2_Game5_O(self):
        
        part_1 = self.P2_Win_Game()**5 * self.P1_Win_Game()**5
        
        part_2 = (self.permutation(5,4)**2 * self.P2_Win_Game()**4 * (1 - self.P2_Win_Game()) *
                  self.P1_Win_Game()**4 * (1 - self.P1_Win_Game()) )
                  
        part_3 = (self.permutation(5,3)**2 * self.P2_Win_Game()**3 * (1 - self.P2_Win_Game())**2 *
                  self.P1_Win_Game()**3 * (1 - self.P1_Win_Game())**2 )
                  
        part_4 = (self.permutation(5,2)**2 * self.P2_Win_Game()**2 * (1 - self.P2_Win_Game())**3 *
                  self.P1_Win_Game()**2 * (1 - self.P1_Win_Game())**3 )
                  
        part_5 = (self.permutation(5,1)**2 * self.P2_Win_Game()**1 * (1 - self.P2_Win_Game())**4 *
                  self.P1_Win_Game()**1 * (1 - self.P1_Win_Game())**4 )
                  
        part_6 = (1 - self.P2_Win_Game())**5 * (1 - self.P1_Win_Game())**5
                  
        return part_1 + part_2 + part_3 + part_4 + part_5 + part_6 
        
    # SCORE 6:6, 7:7, 8:8, ... BOTH PLAYERS WIN 5 GAMES EACH
    def P1_Game_greater_than5_O(self, m):
        self.m = m
        return (2 * self.P1_Win_Game())**(m - 5) * (self.P1_Game5_O())
    
    
    # SYNTHETIC GAMES. SETS WON AT 7:5, 8:6, 9:7, 10:8, ...
    def P1_Game_from_synth(self):
        six_o =  self.P1_Game_greater_than5_O(6) 
        exp1 = self.P1_Win_Game() * (1 - self.P2_Win_Game())
        exp2 = (self.P1_Win_Game() + self.P2_Win_Game()) - 2*self.P1_Win_Game()*self.P2_Win_Game()
        return six_o * (exp1 / exp2)
    
    
    def P1_Set_To_Love_Counter(self):
        # win all 6 games
        return self.P1_Win_Game()**3 * (1 - self.P2_Win_Game())**3 
        
    def P1_Set_6_1_Counter(self):
        
        exp1 = 3*(self.P1_Win_Game())**4 * (1 - self.P2_Win_Game())**2 * self.P2_Win_Game()
        exp2 = 3*(self.P1_Win_Game())**3 * (1 - self.P2_Win_Game())**3 * (1 - self.P1_Win_Game())
        
        return exp1 + exp2

    def P1_Set_6_2_Counter(self):
        
        exp1 = (1 - self.P2_Win_Game() * (self.permutation(4,4)*(self.P1_Win_Game())**4 * 
                self.permutation(3,1) * (1 - self.P2_Win_Game()) * self.P2_Win_Game()**2))
        
        exp2 = (1 - self.P2_Win_Game() * (self.permutation(4,3)*(self.P1_Win_Game())**3 * (1 - self.P1_Win_Game()) *
                self.permutation(3,2) * (1 - self.P2_Win_Game())**2 * self.P2_Win_Game()))
        
        exp3 = (1 - self.P2_Win_Game()* (self.permutation(4,2)*(self.P1_Win_Game())**2 *(1 - self.P1_Win_Game())**2 *
                self.permutation(3,3) * (1 - self.P2_Win_Game())**3))  
        
        return exp1 + exp2 + exp3
    
    
    def P1_Set_6_3_Counter(self):
        
        exp1 = (self.P1_Win_Game() * (self.permutation(4,4)*(self.P1_Win_Game())**4 * 
                self.permutation(4,1) * (1 - self.P2_Win_Game()) * self.P2_Win_Game()**3))
        
        exp2 = (self.P1_Win_Game() * (self.permutation(4,3)*(self.P1_Win_Game())**3 * (1 - self.P1_Win_Game()) *
                self.permutation(4,2) * (1 - self.P2_Win_Game())**2 * self.P2_Win_Game()**2))
        
        exp3 = (self.P1_Win_Game() * (self.permutation(4,2)*(self.P1_Win_Game())**2 * (1 - self.P1_Win_Game())**2 *
                self.permutation(4,3) * (1 - self.P2_Win_Game())**3 * self.P2_Win_Game()))  
        
        exp4 = (self.P1_Win_Game() * (self.permutation(4,1)*(self.P1_Win_Game()) * (1 - self.P1_Win_Game())**3 *
                self.permutation(4,4) * (1 - self.P2_Win_Game())**4))      
        
        return exp1 + exp2 + exp3 + exp4   
        
    def P1_Set_6_4(self):
        
        return (self.P1_Win_Game() **6) * ((1 - self.P1_Win_Game())**4) * self.permutation(9,5) 
    
    def P1_Set_6_4_Counter(self):
        
        exp1 = (1 - self.P2_Win_Game() * (self.permutation(5,5)*(self.P1_Win_Game())**5 * 
                self.permutation(4,0) * self.P2_Win_Game()**4))
        
        exp2 = (1 - self.P2_Win_Game() * (self.permutation(5,4)*(self.P1_Win_Game())**4 * (1 - self.P1_Win_Game()) *
                self.permutation(4,1) * (1 - self.P2_Win_Game()) * self.P2_Win_Game()**3))
        
        exp3 = (1 - self.P2_Win_Game() * (self.permutation(5,3)*(self.P1_Win_Game())**3 * (1 - self.P1_Win_Game())**2 * 
                self.permutation(4,2) * (1 - self.P2_Win_Game())**2 * self.P2_Win_Game()**2))  
        
        exp4 = (1 - self.P2_Win_Game() * (self.permutation(5,2)*(self.P1_Win_Game()**2) * (1 - self.P1_Win_Game())**3 *
                self.permutation(4,3) * (1 - self.P2_Win_Game())**3 * self.P2_Win_Game()))    
        
        exp5 = (1 - self.P2_Win_Game() * (self.permutation(5,1)*(self.P1_Win_Game()) * (1 - self.P1_Win_Game())**4 *
                self.permutation(4,4) * (1 - self.P2_Win_Game())**4))  
        
        return exp1 + exp2 + exp3 + exp4 + exp5     

    # senthetic games 
    def P1_Game_frm_deuce(self):
        six_o =  self.Game_greater_than5_O(6)
        return six_o
    
    def P1_Win_Set_with_total_Game_greater_than11(self, n, m):
        # win set 7:5, 8:6, 9:7, 10:8, ... 
        self.n = n
        self.m = m
        assert type(n) == int, "Use integer values for score!"
        assert type(m) == int, "Use integer values for score!"
        assert n == m + 2, "The difference must be two !"
        return (self.P1_Win_Game())**2 * self.P1_Game_greater_than5_O(m)
   
    def P1_Set_via_Tie_Br(self):
        return self.Tie_Br_win() * (self.Game_greater_than5_O(6))
    
    # In all tie breaks Player1 assumed to start the serving 
    def TieBr7_Love(self):
        # Serve sequence: self.p1 self.p2self.p2 self.p1self.p1 self.p2self.p2 ==> All 7 points won by Player1
        # Player1 self.p1, Player2 self.p2 wins with self.p1 and self.p2 respectively 
        return self.p1**3 * (1 - self.p2)**4

    def TieBr7_1(self):      
        """ Win the last point and six of the previous 7
        From self.p1 self.p2self.p2 self.p1self.p1 self.p2self.p2 win all 3 of own serves and 3 out of 4 opponent's 
        serve AND win all 4 of opponent's serves and 2 out of 3 own serves
        self.p1 * self.permutation(3,3)*self.p1**3 * self.permutation(4,3)*(1-self.p2)**3 * self.p2 + 
        self.permutation(4,3) *
        self.permutation(4,4) * (1-self.p2)**4 * self.permutation(3,2) * self.p1**2 * (1-self.p1) """
        return (4 * self.p1**4 * (1 - self.p2)**3 * self.p2) + (3 * self.p1**3 (1 - self.p2)**4 * (1 - self.p1))

    def TieBr7_2(self):
        
        part_1 = (self.p1*(self.permutation(4,4) * self.p1**4 * self.permutation(4,2)*(1-self.p2)**2 * self.p2**2))
        
        part_2 = (self.p1*(self.permutation(4,3) * self.p1**3 * 
                           (1 - self.p1)**1 * self.permutation(4,3)*(1-self.p2)**3 * self.p2**1))
        
        part_3 = (self.p1*(self.permutation(4,2) * self.p1**2 * 
                           (1 - self.p1)**2 * self.permutation(4,4)*(1-self.p2)**4))
        
        return part_1 + part_2 + part_3
    
    def TieBr7_3(self):
        part_1 = ((1-self.p1)*(self.permutation(5,5) * self.p1**5 * (1 - self.p1)**0 * 
                               self.permutation(4,1)*(1-self.p2)**1 * self.p2**3))
        part_2 = ((1-self.p1)*(self.permutation(5,4) * self.p1**4 * (1 - self.p1)**1 * 
                               self.permutation(4,2)*(1-self.p2)**2 * self.p2**2))
        part_3 = ((1-self.p1)*(self.permutation(5,3) * self.p1**3 * (1 - self.p1)**2 * 
                               self.permutation(4,3)*(1-self.p2)**3 * self.p2**1))
        part_4 = ((1-self.p1)*(self.permutation(5,2) * self.p1**2 * (1 - self.p1)**3 * 
                               self.permutation(4,4)*(1-self.p2)**4 * self.p2**0))
        return part_1 + part_2 + part_3 + part_4

    def TieBr7_4(self):
        part_1 = ((1-self.p1)*(self.permutation(5,5) * self.p1**5 * (1 - self.p1)**0 * 
                               self.permutation(5,1)*(1-self.p2)**1 * self.p2**5))
        part_2 = ((1-self.p1)*(self.permutation(5,4) * self.p1**4 * (1 - self.p1)**1 * 
                               self.permutation(5,2)*(1-self.p2)**2 * self.p2**4))
        part_3 = ((1-self.p1)*(self.permutation(5,3) * self.p1**3 * (1 - self.p1)**2 * 
                               self.permutation(5,3)*(1-self.p2)**3 * self.p2**3))
        part_4 = ((1-self.p1)*(self.permutation(5,2) * self.p1**2 * (1 - self.p1)**3 * 
                               self.permutation(5,4)*(1-self.p2)**4 * self.p2**2))
        part_5 = ((1-self.p1)*(self.permutation(5,1) * self.p1**1 * (1 - self.p1)**4 * 
                               self.permutation(5,5)*(1-self.p2)**5 * self.p2**1))
        return part_1 + part_2 + part_3 + part_4 + part_5   
    
    def TieBr7_5(self):
        part_1 = (self.p1*(self.permutation(5,5) * self.p1**5 * (1 - self.p1)**0 * 
                           self.permutation(6,1)*(1-self.p2)**1 * self.p2**5))
        part_2 = (self.p1*(self.permutation(5,4) * self.p1**4 * (1 - self.p1)**1 * 
                           self.permutation(6,2)*(1-self.p2)**2 * self.p2**4))
        part_3 = (self.p1*(self.permutation(5,3) * self.p1**3 * (1 - self.p1)**2 * 
                           self.permutation(6,3)*(1-self.p2)**3 * self.p2**3))
        part_4 = (self.p1*(self.permutation(5,2) * self.p1**2 * (1 - self.p1)**3 * 
                           self.permutation(6,4)*(1-self.p2)**4 * self.p2**2))
        part_5 = (self.p1*(self.permutation(5,1) * self.p1**1 * (1 - self.p1)**4 * 
                           self.permutation(6,5)*(1-self.p2)**5 * self.p2**1))
        part_6 = (self.p1*(self.permutation(5,0) * self.p1**0 * (1 - self.p1)**5 * 
                           self.permutation(6,6)*(1-self.p2)**6 * self.p2**0))  
        return part_1 + part_2 + part_3 + part_4 + part_5 + part_6 
        
    def TieBr6_6(self):
        part_1 = (self.permutation(6,6) * self.p1**6 * (1 - self.p1)**0 * 
                  self.permutation(6,0)*(1-self.p2)**0 * self.p2**6)
        part_2 = (self.permutation(6,5) * self.p1**5 * (1 - self.p1)**1 * 
                  self.permutation(6,1)*(1-self.p2)**1 * self.p2**5)
        part_3 = (self.permutation(6,4) * self.p1**4 * (1 - self.p1)**2 * 
                  self.permutation(6,2)*(1-self.p2)**2 * self.p2**4)
        part_4 = (self.permutation(6,3) * self.p1**3 * (1 - self.p1)**3 * 
                  self.permutation(6,3)*(1-self.p2)**3 * self.p2**3)    
        part_5 = (self.permutation(6,2) * self.p1**2 * (1 - self.p1)**4 * 
                  self.permutation(6,4)*(1-self.p2)**4 * self.p2**2)
        part_6 = (self.permutation(6,1) * self.p1**1 * (1 - self.p1)**5 * 
                  self.permutation(6,5)*(1-self.p2)**5 * self.p2**1)
        part_7 = (self.permutation(6,0) * self.p1**0 * (1 - self.p1)**6 * 
                  self.permutation(6,6)*(1-self.p2)**6 * self.p2**0)
        return part_1 + part_2 + part_3 + part_4 + part_5 + part_6 + part_7 
    
    def P1_TieBr_from_deuce(self):
        return (self.p1 * (1- self.p2)) / (self.p1 + self.p2 - 2*self.p1*self.p2)
    
    def P1_TieBr7_6(self):
        return self.TieBr_from_deuce() * self.TieBr6_6()

    def P1_Tie_Br_win(self):
        
        return (self.TieBr7_Love() + self.TieBr7_1() + self.TieBr7_2() + self.TieBr7_3() 
                + self.TieBr7_4() + self.TieBr7_5() + self.TieBr7_6())

    


# In[ ]:


class test_tb(Prob_Player_Wins_points):
    # In all tie breaks Player1 assumed to start the serving 
    def P1_TieBr7_Love(self):
        return self.permutation(7,7) * self.p1**7

    def P1_TieBr7_1(self):
        return self.permutation(7,6) * (self.p1**7) * (1 - self.p1)

    def P1_TieBr7_2(self):
        return self.permutation(8,6) * (self.p1**7) * (1 - self.p1)**2
    
    def P1_TieBr7_3(self):
        return self.permutation(9,6) * (self.p1**7) * (1 - self.p1)**3

    def P1_TieBr7_4(self):
        return self.permutation(10,6) * (self.p1**7) * (1 - self.p1)**4 
    
    def P1_TieBr7_5(self):
        return self.permutation(11,6) * (self.p1**7) * (1 - self.p1)**5
        
    def P1_TieBr6_6(self):
        return self.permutation(12,6) * (self.p1**6) * (1 - self.p1)**6
    
    def P1_TieBr7_6(self):
        return self.P1_TieBr6_6() * self.P1_From_Deuce()   
    
    def P1_win_set(self):
        return (self.P1_Set_Love() + self.P1_Set_6_1() + self.P1_Set_6_2() + self.P1_Set_6_3() + 
                self.P1_Set_6_4() + self.P1_Set_7_5() + self.P1_Set_7_6(6))
    
    def Tie_Br_win(self):
        # add up all tie break prob
        return (self.P1_TieBr7_Love() + self.P1_TieBr7_1() + self.P1_TieBr7_2() + self.P1_TieBr7_3() +
                self.P1_TieBr7_4() + self.P1_TieBr7_5() + self.P1_TieBr7_6())
    
tt = test_tb(0.62, 0.60)
tt.P1_TieBr7_1()

