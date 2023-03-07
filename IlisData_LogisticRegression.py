
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = datasets.load_iris()


# 데이터셋
input_data = data['data']      # 꽃의 특징 (input data)
target_data = data['target']   # 꽃 종류를 수치로 나타낸 것(0~2) (target data)
flowers = data['target_names'] # 꽃 종류를 이름으로 나타낸 것
feature_names = data['feature_names']   # 꽃 특징들의 명칭
# sepal : 꽃받침
# petal : 꽃잎

print('꽃을 결정짓는 특징 : {}'.format(feature_names))
print('꽃 종류 : {}'.format(flowers))


# pandas 사용하여 데이터 출력 해보기
iris_df = pd.DataFrame(input_data, columns=feature_names)
iris_df['species'] = target_data

print(iris_df.head(10))   # 맨 위에 있는 데이터 10개 출력
print(iris_df.describe()) # 데이터의 정보 출력


# 데이터 개괄적 특징 파악
sns.pairplot(iris_df, hue='species', vars=feature_names)

plt.show()
plt.close()


# 훈련 데이터와 테스트 데이터 분리
train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, random_state=42)

# 표준 점수로 데이터 스케일링
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)

lr = LogisticRegression(max_iter=1000)

# 로지스틱 회귀 학습
lr.fit(train_scaled, train_target)

# 테스트 데이터 예측
pred = lr.predict(test_scaled[:5])
print(pred)

# 로지스틱 회귀 모델의 가중치와 절편
# 다중 분류 가중치와 절편을 출력하면, 각 클래스마다의 가중치 정편을 출력한다.
print(lr.coef_, lr.intercept_)

# 결정 함수(decision_function)로 z1 ~ z3의 값을 구한다.
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

# 소프트맥스 함수를 사용한 각 클래스들의 확률
from scipy.special import softmax

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))

# https://itstory1592.tistory.com/11