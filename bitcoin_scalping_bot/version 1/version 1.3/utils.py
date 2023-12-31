import pandas as pd
import matplotlib.pyplot as plt
import os

def pnl_graph(num):
    df = pd.read_csv("bitcoin_scalping_bot\\version_3\\train_log\\log_{}.csv".format(str(num)), index_col=0)

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
    df = pd.read_csv("bitcoin_scalping_bot\\version_3\\train_log\\log_{}.csv".format(str(num)), index_col=0)
    action_value_counts = df['action'].value_counts()

    plt.figure(figsize=(4, 4))
    plt.bar(action_value_counts.index, action_value_counts.values)
    plt.title("Frequency of 0 to 10 in 'action'")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.xticks(range(11))
    plt.show()
    
def reward_graph(num):
    df = pd.read_csv("bitcoin_scalping_bot\\version_3\\train_log\\log_{}.csv".format(str(num)), index_col=0)

    df['reward_sign'] = df['reward'].apply(lambda x: 1 if x > 0 else -1)

    reward_value_counts = df['reward_sign'].value_counts()

    plt.figure(figsize=(4, 4))
    plt.bar(reward_value_counts.index, reward_value_counts.values, color=['red', 'blue'])
    plt.title("Frequency of -1 and 1 in 'reward'")
    plt.xlabel("reward_sign")
    plt.ylabel("Frequency")
    plt.xticks([-1, 1])
    plt.show()    