from collections import deque
from datetime import datetime


class Environment5:
    def __init__(self, chart_data=None, raw_data=None,chart_index:list = None, risk_adverse= 1.3 ,balance=100000000, transaction=0.0004, max_leverage=10):
        self.chart_data = chart_data # numpy array
        self.raw_data = raw_data
        self.chart_index = chart_index # python list
        self.idx = 0

        self.risk_adverse = risk_adverse # 손실에 주는 가중치 
        self.transaction = transaction # 거래수수료
        self.max_leverage = max_leverage
        
        self.current_state = self.chart_data[self.idx]
        self.next_state = self.chart_data[self.idx+1]
                    
        self.current_price = self.raw_data["mid_price"].iloc[self.idx]
        self.next_price = self.raw_data["mid_price"].iloc[self.idx+1]
        
        self.initial_balance = balance
        self.balance = [balance]  # 포트폴리오가 보유한 현금
        self.bitcoin = [0]  # 포트폴리오가 보유한 비트코인의 가치 (매 거래마다 바로 청산됨)
        self.portfolio_value = []
        
        self.action_info = [-1, 0, 1]
        self.action_size = len(self.action_info)
        self.action_list = deque([len(self.action_info)//2 for i in range(32)]) # 이전 32분을 저장함.

        self.position = 0 # 이전의 position 비율을 저장하는 변수. (+)는 long, (-)는 short
        self.position_list = deque([0 for i in range(32)]) # 이전 32분을 저장함.
        
        # self.profit_queue = deque([0.0004 for i in range(64)]) # 이전 1시간의 변동성을 고려함
        
        
    def reset(self):
        self.idx = 0
        state = self.chart_data[self.idx]
        return state
    
    def get_profit_std(self, profit):
        self.profit_queue.popleft()
        self.profit_queue.append(profit)
        std = (pow(sum(self.profit_queue),2)/len(self.profit_queue))**0.5
        return abs(std)
    
    
    def step(self, action):
        self.current_state = self.chart_data[self.idx]
        self.next_state = self.chart_data[self.idx+1]                    
        self.current_price = self.raw_data["mid_price"].iloc[self.idx]
        self.next_price = self.raw_data["mid_price"].iloc[self.idx+1]

        current_value = self.balance[-1] + self.bitcoin[-1]*self.current_price 
        self.portfolio_value.append(current_value)
        
        s_prime = self.chart_data[self.idx+1]
        
        # action list에 새로운 action 추가해줌
        self.action_list.append(action)
        self.action_list.popleft()
            
        # reward 계산
        profit = self.get_reward(action)
        
        # 얻은 수익률의 표준편차를 구해준다.
        # std = self.get_profit_std(profit)  
            
        # sharpe ratio를 maximize하는 형식
        reward = profit
        
        # risk adverse정도를 고려해서 reward 계산
        if reward<0:
            reward =  reward * self.risk_adverse
        
        current_time = self.raw_data.iloc[self.idx].name
        current_day = datetime.strftime(current_time, '%Y-%m-%d')
        next_time = self.raw_data.iloc[self.idx+1].name
        next_day = datetime.strftime(next_time, '%Y-%m-%d')

        # 시간 index 갱신
        self.idx += 1

        if current_day == next_day:
            return {"state_time":current_day, 
                    "next_state":s_prime, 
                    "reward":round(reward,8), 
                    "done":False, 
                    "portfolio_value":self.portfolio_value[-1], 
                    "balance":self.balance[-1], 
                    "bitcoin":self.bitcoin[-1], 
                    "position_list":self.position_list,
                    "action_list":self.action_list,
                    "current_price": self.current_price,
                    "index": self.idx}
        else:
            print(f'{current_day}에서 {format(round(self.portfolio_value[-1]), ",")}원으로 trading stop, {round((self.portfolio_value[-1]/100000000-1)*100,1)}%')
            print("#########################################################################")
            
            P = self.portfolio_value[-1]
            B = self.balance[-1]
            C = self.bitcoin[-1]
            
            # 다음날로 넘어갈 때 포지션 청산 
            self.position = 0
            
            # 다음날 거래는 다시 1억으로 시작
            self.balance = [self.balance[0]]  # 포트폴리오가 보유한 현금
            self.bitcoin = [0]  # 포트폴리오가 보유한 비트코인의 가치 (매 거래마다 바로 청산됨)
            self.portfolio_value = []
            
            return {"state_time":current_day, 
                    "next_state":s_prime, 
                    "reward":round(reward,8), 
                    "done":True, 
                    "portfolio_value":P, 
                    "balance":B, 
                    "bitcoin":C, 
                    "position_list":self.position_list,
                    "action_list":self.action_list,
                    "current_price": self.current_price,
                    "index": self.idx}
    
    
    
    def position_calc(self, action): # max_leverage를 고려해서 position을 계산해주는 함수
        action = self.action_info[action] # action의 실제 action
        execute_position = self.position
        
        if action * self.position > 0:
            if  self.position + action > self.max_leverage:
                ratio = 0
                self.position = self.max_leverage
                return self.position, ratio, False
            
            elif self.position + action < -self.max_leverage:
                ratio  = 0 
                self.position = -self.max_leverage
                return self.position, ratio, False
            else:
                self.position += action
                ratio = action
                return self.position, ratio, False
                    
        elif action * self.position < 0:
            self.position = action 
            ratio = action 
            return self.position, ratio, True
        
        elif action*self.position==0:
            if action==0:
                self.position = action 
                return self.position, 0, False
            
            elif self.position==0:
                self.position = action 
                return self.position, self.position, False
            
        
    def get_reward(self, action):     
        pre_position = self.position
        
        
        # Short
        if action < (self.action_size//2):
            position, ratio, execution = self.position_calc(action)    
            
            self.position_list.append(position)
            self.position_list.popleft()

            sell_budget = self.initial_balance * ratio
            if execution: 
                # 직전 포지션이 long, clearing_budget: (+)
                clearing_budget = (self.bitcoin[-1]) * self.current_price
                # 이전에 매수를 한 경우 => 현재 매도(long 청산)
                self.balance.append(self.balance[-1] - sell_budget*(1-self.transaction) + clearing_budget*(1-self.transaction)) # clearing budget logic이 오류가 있음.
                self.bitcoin.append(sell_budget/self.current_price)          
            else: # 직전 포지션이 short
                self.balance.append(self.balance[-1] - sell_budget*(1-self.transaction))
                self.bitcoin.append(self.bitcoin[-1] + sell_budget/self.current_price)  
        
            current_value = self.portfolio_value[-1]
            next_value = self.balance[-1] + self.bitcoin[-1]*(self.next_price)
            reward = (next_value - current_value) / self.initial_balance
            reward = 1 if reward >= 0 else -1
            return reward
            
        # Long
        elif action > (self.action_size//2):
            position, ratio, execution = self.position_calc(action)
            
            self.position_list.append(position)
            self.position_list.popleft()

            buy_budget = self.initial_balance * ratio
    
            if execution: # 직전 포지션 short | clearing_budget: (-) 
                clearing_budget = (self.bitcoin[-1]) * self.current_price
                self.balance.append(self.balance[-1] - buy_budget*(1+self.transaction) + clearing_budget*(1+self.transaction)) 
                self.bitcoin.append(buy_budget/self.current_price)              

            else: # 직전 포지션 long
                self.balance.append(self.balance[-1] - buy_budget*(1+self.transaction))
                self.bitcoin.append(self.bitcoin[-1] + buy_budget/self.current_price)  
            
            current_value = self.portfolio_value[-1]
            next_value = self.balance[-1] + self.bitcoin[-1]*(self.next_price)
            reward = (next_value - current_value) / self.initial_balance
            reward = 1 if reward >= 0 else -1
            return reward
        
        # 청산
        elif action == (self.action_size//2):
            self.position_list.append(0)
            self.position_list.popleft()
            
            if pre_position>=0:
                clearing_budget = (self.bitcoin[-1]) * self.current_price
                self.balance.append(self.balance[-1] + clearing_budget*(1-self.transaction))
                self.bitcoin.append(0)
                
            elif pre_position<0: 
                clearing_budget = (self.bitcoin[-1]) * self.current_price
                self.balance.append(self.balance[-1] + clearing_budget*(1+self.transaction))
                self.bitcoin.append(0)
            
            return -0.05
