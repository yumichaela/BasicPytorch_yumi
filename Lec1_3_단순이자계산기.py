# 이자 계산기를 만들자. 
# 원금, 이자율, 기간 
# 변수, 메소드 

'''
- 원금(principal) : 외부에서 받을 것 
- 이자율(interest_rate) : 생성할 때 초기화 (초기값을 가지고 있을 것) -> 추후에 수정 가능하게 만들 것 
- 기간(period) : 외부에서 받을 것 

함수들
- _set_interest : 이자를 계산해서 self로 가지고 있기 
- get_interest : 원금이랑 기간을 입력하면 이자를 출력 -> _set_interest를 불러야 함 
- change_interest_rate : 새로운 이자율 받아서 변수를 바꾸기 
'''

class Calculator:
    def __init__(self, interest_rate):
        self.interest_rate = interest_rate * 0.01
    
    def _set_interest(self, principal, period):
        self.interest = principal * self.interest_rate * period 
        
    def get_interest(self, principal, period):
        self._set_interest(principal, period)
        return self.interest
        
    def change_interest_rate(self, new_interest_rate):
        self.interest_rate = new_interest_rate * 0.01
        
    def interest_print(self, principal):
        print("원금 {}원, 이자율 {:.2f}%, 이자 {}원".format(principal, self.interest_rate, int(self.interest)))
        
interest_rate = 5.1 # (%) 
principal = 1000000 
period = 10 # 개월
MyCal = Calculator(interest_rate)
interest = MyCal.get_interest(principal, period) 
MyCal.interest_print(principal)

new_interest_rate = 3.1 # (%) 
MyCal.change_interest_rate(new_interest_rate)
interest = MyCal.get_interest(principal, period) 
MyCal.interest_print(principal)
