# ShiftedScaleQuantization


#TODO
IMAGE net setup

Multi bit scaling setup

3,4 bit test

90.47


# Abstract
# Introduction
- pruning/low-rank/kd등 일반적인 경량화 방법
- Quantization 소개와 장점
- Quantization 종류(PTQ/QAT)
- Weight 차이 감소 vs Loss 감소
-----
- 우리가 할것 (shifting)
(장점)
- 기본적인 HW 구조를 유지한채 할 수 있다.
-- structure 유지(permute)
-- fully 사용 가능(full bit)
- 작은 computation cost만 증가한다.
- 
(단점)
- 상위 비트 제약? 이건 여러개의 scale을 두어 해결해보자.
