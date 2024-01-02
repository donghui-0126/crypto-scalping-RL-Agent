# Cryrto scalping RL Agent
# **프로젝트 목표**
Binance api를 연결후 BTC-USTD 선물거래
<br>

## **프로젝트 주제**
## **강화학습을 이용한 크립토 단타매매 봇**
강화학습을 사용해서 Sclaping을 진행하는 단타매매 봇입니다. <br>
최종 목표는 binance api에 연결을해서 선물거래를 진행하는 것입니다.

이 프로젝트는 몇가지 상상에서 시작됐습니다. 

결국 가격이라는 것은 시장미시적인 거래와 체결로부터 산출되는 가격이라고 생각합니다. 
그리고 가격을 관측하는 간격이 짧아지면 짧아질 수록 이 가정은 더욱 잘 적용된다고 생각합니다. 

또한, 짧은 시간의 거래에서 관측되는 거래 참여자 들은 대부분 Rule-Base 기반의 자동매매봇이라고 생각했습니다.<br>

저는 여기서 Reinforcement Learning가 적용될 수 있다고 생각했습니다. <br>
Rule-Base 모델을 Rule-base기반 모델이 생성한 데이터로 학습된 Data-Base 모델로 이길 수 있다고 생각했습니다. 

-------
### **데이터**
데이터는 upbit의 비트코인 1분봉 데이터를 사용했습니다. <br>
입력데이터는 지난 60분동안의 가격과 기술적지표에 대한 데이터를 사용합니다.<br>
각 데이터는 원본 가격데이터와 변화율 데이터를 사용합니다.  

-------

### **프로젝트 가정**
- 학습에는 upbit 1분봉 데이터를 사용하지만 long과 short 모두를 가능하다고 가정했습니다.<br>
- levearge 설정이 자유롭게 가능하다고 설정했습니다.<br>
- 매 1분마다 agent가 매수/매도 주문을 넣습니다. <br>
- 가격은 **(high+low)/2**로 체결된다는 가정을 설정했습니다. <br>
- Agent가 매수/매도 주문을 넣고 다음 1분안에 모든 주문이 체결된다는 가정을 설정했습니다. <br>


### **버전 정보**
* version 1.1: DNN사용, 단순히 가격데이터만 사용함.
* version 1.2: CNN사용, 기술적 지표도 사용함. 
* version 1.3: version_2에서 action을 단순화함.
* version 1.4: entropy term을 loss값에 추가. batch nomalization 사용.

* **version 2.0: Agent 방향성 변경** 
  * **version 2.0 algorithm** 
    1. Select Strategy Duration
    2. Simulate Strategy N times (Reward = Sortino Ratio)
    3. Get Reward that Expectation of N times simulated Reward

#### Architecture: [link](https://github.com/donghui-0126/crypto-scalping-RL-Agent/blob/main/Scalping%20bot.svg)

-------


### **프로젝트 기간**
* Project Period : 2023/10 ~~ (진행중)

<br>

#### 추후 개선방안: VPIN과 같은 지표 사용. 기술적지표 추가. Data 정규화.  


