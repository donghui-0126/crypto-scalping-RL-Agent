import torch
from torch.distributions import Categorical 
from env4 import Environment4
from ppo4 import PPO4
import tqdm
import pandas as pd
import numpy as np
import warnings 
import pickle
import os
from utils import data_preprocess

warnings.filterwarnings('ignore')

T_horizon = 64

date_log = []
value_log = []
action_log = []
reward_log = []

def main(model_name, risk_adverse, epochs = 100, transaction=0.0002, max_leverage=10, num_action=5):
    with open('C:\\Users\\user\\Documents\\GitHub\\crypto-scalping-RL-Agent\\data\\df_final.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('C:\\Users\\user\\Documents\\GitHub\\crypto-scalping-RL-Agent\\data\\df_final_raw.pkl', 'rb') as f:
        df_raw = pickle.load(f)    
    print("데이터 적재 완료...")
    
    df_final, df_raw = data_preprocess(df, df_raw, sequence_length=64)
    print("데이터 전처리 완료...")
    
    if model_name=="ppo4":
        model = PPO4(learning_rate=0.001, eps_clip=0.1, K_epoch=3)

    
    env = Environment4(df_final, df_raw, risk_adverse = risk_adverse, transaction=transaction, max_leverage=max_leverage)
    state = env.reset()
        
    h1_out = (torch.zeros([1, 1, 64], dtype=torch.float), torch.zeros([1, 1, 64], dtype=torch.float))
    
        
    for n_epi in tqdm.tqdm(range(epochs)):
        print(" # of episode :{} ".format(n_epi+1))
        #position_action 이라는 list를 만들어주기위한 임시 변수
        temp_position = [0 for x in range(32)] 
        temp_action = [num_action//2 for x in range(32)]
        temp_action_position = np.array(temp_action+temp_position)

        done = False   
        log_date_list = []
        log_value_list = []
        log_action_list = []
        log_reward_list = []
        log_balance_list = []
        log_coin_list = []
        log_current_price_list = []
        log_index_list = []
                 
        
        while not done:
            for t in range(T_horizon):
                h1_in = h1_out
                
                prob, h1_out = model.pi(torch.from_numpy(state).float(), h1_in, torch.tensor(temp_action_position))
                
                prob = prob.view(-1)
                
                action_distribition = Categorical(prob)
                
                action = action_distribition.sample().item()
                env_step_dict = env.step(action)
                
                state_time = env_step_dict["state_time"]
                next_state = env_step_dict["next_state"]
                reward = env_step_dict["reward"]
                done = env_step_dict["done"]  
                portfolio_value = env_step_dict["portfolio_value"]  
                balance = env_step_dict["balance"]  
                bitcoin = env_step_dict["bitcoin"]
                position_list = env_step_dict["position_list"]
                action_list = env_step_dict["action_list"]  
                current_price = env_step_dict["current_price"]
                index = env_step_dict["index"]
                

                # log 저장
                log_date_list.append(state_time)
                log_value_list.append(portfolio_value)
                log_action_list.append(action)
                log_reward_list.append(reward)
                log_balance_list.append(balance)
                log_coin_list.append(bitcoin)
                log_current_price_list.append(current_price)
                log_index_list.append(index)

                
                # position action 병합
                action_list = list(action_list)
                position_list = list(position_list)

                action_position_list = action_list + position_list
                
                temp_action_position = action_position_list
                
                model.put_data([np.array(state, dtype=np.float32),
                            action, 
                            reward,
                            np.array(next_state, dtype=np.float32),\
                            prob[action].item(),\
                            h1_in, h1_out, done, action_position_list])

                state = np.array(next_state, dtype=np.float32)

                if done==True:
                    break
                
            model.train_net()

            log = pd.DataFrame([log_date_list, log_value_list, log_action_list, log_reward_list, log_balance_list, log_coin_list, position_list, log_current_price_list, log_index_list]).T
            log.columns = ["date", "value", "action", "reward", "balance", "coin", "position","price", "index"]
            log.to_csv("bitcoin_scalping_bot\\version_4\\train_log\\log_{}.csv".format(n_epi+1))
            
    print("#####")
    print("END")
    print("#####")
    

if __name__ == '__main__':
    filepath = "bitcoin_scalping_bot\\version_4\\train_log"
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)
    main(model_name="ppo4", risk_adverse=2, epochs=300, transaction=0.004, max_leverage=10)