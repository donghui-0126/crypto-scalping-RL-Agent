import torch
from torch.distributions import Categorical 
from env4 import Environment4
from ppo4 import PPO4
import tqdm
import pandas as pd
import numpy as np
import warnings 
import pickle
warnings.filterwarnings('ignore')

T_horizon = 128

date_log = []
value_log = []
action_log = []
reward_log = []

def main(model_name, risk_adverse, epochs = 100, transaction=0.0002, max_leverage=10):
    with open('bitcoin_scalping_bot\\upbit_data\\train_data_2023_3D.pkl', 'rb') as f:
        df = pickle.load(f)
    df_index = pd.read_csv("bitcoin_scalping_bot\\upbit_data\\train_data_2023_3D_index.csv", index_col=0)
    
    print("데이터 적재 완료...")
    if model_name=="ppo4":
        model = PPO4()

    env = Environment4(df, list(df_index.index), risk_adverse = risk_adverse, transaction=transaction, max_leverage=max_leverage)
    state = env.reset()
        
    h1_out = (torch.zeros([1, 1, 64], dtype=torch.float), torch.zeros([1, 1, 64], dtype=torch.float))
    
        
    for n_epi in tqdm.tqdm(range(epochs)):
        #position_action 이라는 list를 만들어주기위한 임시 변수
        temp = [5 for x in range(31)] 
        temp.append(0)
        position_action = np.array(temp)
        
        done = False   
        date_list = []
        value_list = []
        action_list = []
        reward_list = []
        balance_list = []
        coin_list = []
        position_list = []
        all_action_list  =[]
        current_price_list = []
        index_list = []         
        while not done:
            for t in range(T_horizon):
                h1_in = h1_out
                
                prob, h1_out = model.pi(torch.from_numpy(state).float(),h1_in, torch.tensor(position_action))
                
                prob = prob.view(-1)
                
                action_distribition = Categorical(prob)
                
                action = action_distribition.sample().item() # softmax에서 0~11의 값을 return
                env_step_dict = env.step(action)
                
                state_time = env_step_dict["state_time"]
                next_state = env_step_dict["next_state"]
                reward = env_step_dict["reward"]
                done = env_step_dict["done"]  
                portfolio_value = env_step_dict["portfolio_value"]  
                balance = env_step_dict["balance"]  
                bitcoin = env_step_dict["bitcoin"]
                position = env_step_dict["position"]  
                all_action = env_step_dict["action_list"] 
                current_price = env_step_dict["current_price"]
                index = env_step_dict["index"]
                

                # log 저장
                date_list.append(state_time)
                value_list.append(portfolio_value)
                action_list.append(action)
                reward_list.append(reward)
                balance_list.append(balance)
                coin_list.append(bitcoin)
                position_list.append(position)
                all_action_list.append(all_action)
                current_price_list.append(current_price)
                index_list.append(index)

                
                # position action 병합
                position_action = list(all_action)
                position_action.append(position)

                model.put_data([np.array(state, dtype=np.float32),
                            action, 
                            reward,
                            np.array(next_state, dtype=np.float32),\
                            prob[action].item(),\
                            h1_in, h1_out, done, position_action])

                state = np.array(next_state, dtype=np.float32)

                if done==True:
                    break
                
            model.train_net()

            log = pd.DataFrame([date_list, value_list, action_list, reward_list, balance_list, coin_list, position_list, current_price_list, index_list]).T
            log.columns = ["date", "value", "action", "reward", "balance", "coin", "position","price", "index"]
            log.to_csv("bitcoin_scalping_bot\\version_4\\train_log\\log_{}.csv".format(n_epi+1))
            
            if done == True:
                print("# of episode :{} ".format(n_epi+1))
                break
            
    print("#####")
    print("END")
    print("#####")
    

if __name__ == '__main__':
    main(model_name="ppo4", risk_adverse=3, epochs=300, transaction=0.004, max_leverage=10)