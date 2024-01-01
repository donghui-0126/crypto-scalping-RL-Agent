import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def pnl_graph(num):
    df = pd.read_csv("bitcoin_scalping_bot\\version_4\\train_log\\log_{}.csv".format(str(num)), index_col=0)

    plt.figure(figsize=(12, 6))

    # Plot the first graph (portfolio value)
    df['value'].plot(label="Portfolio Value")
    plt.xlabel("Min")
    plt.ylabel("Value")

    # Plot the second graph (Bitcoin value)
    (df['price']/df.loc[0,"price"]*100000000).plot(label="Bitcoin Value")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.title("Portfolio and Bitcoin Value")
    plt.grid(True)

    # Show a legend to differentiate the two lines
    plt.legend()

    # Show the combined plot
    plt.show()


def action_graph(num):
    df = pd.read_csv("bitcoin_scalping_bot\\version_4\\train_log\\log_{}.csv".format(str(num)), index_col=0)
    action_value_counts = df['action'].value_counts()

    plt.figure(figsize=(4, 4))
    plt.bar(action_value_counts.index, action_value_counts.values)
    plt.title("Frequency of 0 to 10 in 'action'")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.xticks(range(11))
    plt.show()
    
def reward_graph(num):
    df = pd.read_csv("bitcoin_scalping_bot\\version_4\\train_log\\log_{}.csv".format(str(num)), index_col=0)

    df['reward_sign'] = df['reward'].apply(lambda x: 1 if x > 0 else -1)

    reward_value_counts = df['reward_sign'].value_counts()

    plt.figure(figsize=(4, 4))
    plt.bar(reward_value_counts.index, reward_value_counts.values, color=['red', 'blue'])
    plt.title("Frequency of -1 and 1 in 'reward'")
    plt.xlabel("reward_sign")
    plt.ylabel("Frequency")
    plt.xticks([-1, 1])
    plt.show()    
    
    
def data_preprocess(df, df_raw, sequence_length=64):
    # 주어진 데이터프레임을 슬라이싱
    n_samples = len(df) - sequence_length + 1

    # Extract the data as a NumPy array
    data_array = df.to_numpy()

    # Create an empty array to store the input sequences
    input_sequences = np.empty((n_samples, (df.shape[1]), sequence_length))

    for i in tqdm(range(n_samples)):
        input_seq = data_array[i:i + sequence_length, :].T  # Transpose to have (24, 60) shape
        input_sequences[i] = input_seq
        
    df_raw = df_raw.iloc[sequence_length-1:]
    
    print(input_sequences.shape, df_raw.shape)
    return input_sequences, df_raw