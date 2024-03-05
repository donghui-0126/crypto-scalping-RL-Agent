# Crypto Scalping RL Agent

## **Project Objectives**
Connect to the Binance API and implement BTC-USDT futures trading.

## **Project Theme**
## **Reinforcement Learning-Based Crypto Scalping Bot**
A short-term trading bot for crypto scalping using reinforcement learning. The ultimate goal is to connect to the Binance API and execute futures trading.


### **Data**
The data used is 1-minute candlestick data for Bitcoin from Upbit. Input data includes price and technical indicator data for the last 60 minutes. Each data point utilizes the original price data and its rate of change.

-------

### **Project Assumptions**
- The training uses Upbit's 1-minute candlestick data, assuming the possibility of both long and short positions.
- Leverage settings are flexible.
- The agent places buy/sell orders every minute.
- The price is assumed to be the **(high + low)/2** at which trades are executed.
- The assumption is made that after the agent places buy/sell orders, all orders are executed within the next minute.

### Version 1.0
* Version 1.1: Uses DNN, only utilizes price data.
* Version 1.2: Uses CNN, includes technical indicators.
* Version 1.3: Simplifies actions in version_2.
* Version 1.4: Adds entropy term to the loss value, uses batch normalization.

### **Version 3.0: 
* Version 3.0: use Stable-baseline3/Gymnasium (Ongoing)

### **Project Duration**
* Project Period: 2023/10 ~~ 

<br>

#### Future Improvements: Use indicators like VPIN, add technical indicators, normalize data.
