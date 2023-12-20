# Cryrto scalping RL Agent

<br>

# **프로젝트 주제**
**강화학습을 이용한 크립토 단타매매 봇**
강화학습을 사용해서 Sclaping을 진행하는 단타매매 봇입니다. <br>
최종 목표는 binance api에 연결을해서 선물거래를 진행하는 것입니다.

# **데이터**
데이터는 upbit의 비트코인 1분봉 데이터를 사용했습니다. <br>
입력데이터는 지난 60분동안의 가격과 기술적지표에 대한 데이터를 사용합니다.<br>
각 데이터는 원본 가격데이터와 변화율 데이터를 사용합니다.  

# **프로젝트 가정**
학습에는 upbit 1분봉 데이터를 사용하지만 long과 short 모두를 가능하다고 가정했습니다.<br>
levearge 설정이 자유롭게 가능하다고 설정했습니다.<br>
매 1분마다 agent가 매수/매도 주문을 넣습니다. <br>
가격은 **(high+low)/2**로 체결된다는 가정을 설정했습니다. <br>
Agent가 매수/매도 주문을 넣고 다음 1분안에 모든 주문이 체결된다는 가정을 설정했습니다. <br>


# **버전 정보**
* version_1: DNN사용, 단순히 가격데이터만 사용함.
* version_2: CNN사용, 기술적 지표도 사용함. 
* version_3: version_2에서 action을 단순화함.
* version_4: entropy term을 loss값에 추가. batch nomalization 사용.
* 추후 개선방안: VPIN과 같은 지표 사용. 기술적지표 추가. Data 정규화.  
<br>

# ** 프로젝트 기간**
* Project Period : 2023/10 ~~ (진행중)

<br>