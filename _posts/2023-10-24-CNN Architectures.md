---
title:  "[CS231n] 7.CNN Architectures"
excerpt: "CS231n Lecture Note"

categories:
  - Computer Vision
tags:
  - CS231n
toc: true
toc_sticky: true
last_modified_at: 2023-10-24T08:06:00-05:00
---

# LeNet-5

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/e812544a-0904-4e8c-b46d-bed8bee9d361" height="80%" width="80%"></p>

그림에서 C는 Conv, S는 Subsampling(Pooling)이다. Conv, Pooling이 반복되다가 마지막에 FC를 거치고 output을 내놓는걸 확인할 수 있다. Filter는 Conv의 경우 5X5에 Stride는 1이고 Pooling의 경우는 2X2에 Stride는 2이다. 
이에 따라 처음에 32X32 input을 받고 Filter를 거쳐서 $(32-5)/1 + 1 = 28$로 28X28의 결과가 나오는 것을 확인할 수 있다. 
또한 Pooling 시에는 $(28 - 2) / 2 + 1 = 14$가 된다. 

`Conv - Pool - Conv - Pool - Conv - FC`


# AlexNet

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/47cb8fe9-dc42-4257-90f0-0f05100d3bdc" height="80%" width="80%"></p>

AlexNet는 input으로 227X227X3 size의 이미지를 받고 위 그림에서의 224는 오타이다. 그림 또한 원래 논문에서 사진이 조금 짤렸는데 위쪽 stream과 아래쪽 stream 이렇게 2개로 구성되어 있다. 
이렇게 2개의 stream으로 나누어서 설계한 이유는 당시 GPU의 성능이 좋지 못했기 때문에 2개의 GPU를 활용해 학습하였다. 현재는 하나로 GPU로 연산이 가능하다. 

AlexNet의 First layer를 보면 96개의 11X11 크기의 4 stride Filter를 거치게 된다. 따라서 First layer를 거친 후의 output size는 $(227 - 11)/4 + 1 = 55$가 되어 55X55X96이 된다. 
또한 전체 파라미터의 수는 input 의 depth가 3이었고, 11X11 Filter가 96개 있었으므로 $(11*11*3)*96 = 35k$가 된다. 

다음으로 Pooling layer의 경우 3X3에 stride는 2이다. 따라서 Pooling을 거치게 되면 마찬가지로 $(55 - 3)/2 + 1 =27$이 되어 27X27X96이 된다. 이때 Pooling layer에서 depth는 변하지 않는다.
또한 Pooling layer에서는 파라미터가 없으므로 0이다. 파라미터는 Conv에서만 존재한다.  

최종적으로 마지막까지 계산하한 내용과 AlexNet에 대해 정리한 내용은 아래와 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/67bf4ab2-1e56-4f85-b80b-fc39ed3745ea" height="70%" width="70%"></p>

참고로 Normalization layer는 현재 효용이 없다고 판단되어 사용되지 않는다. 과정을 보면 layer를 거칠수록 size는 점점 작아지는데 반해 filter의 수는 늘어나는 것을 볼 수가 있고, 
마지막 FC7 layer는 일반적으로 통칭되는 용어이기도 한데 FC7 layer는 classifier 직전의 layer를 보통 FC7 layer라고 한다. 


# ZFNet

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/b90290b4-b39b-420b-b7d5-dd28b39d6857" height="80%" width="80%"></p>

ZFNet은 기본적으로 AlexNet와 거의 유사한데 차이가 있다면 AlexNet에서는 Conv1 layer에서 11X11 stride 4의 Filter를 사용했는데 이 Filter가 너무 크다고 판단되어서 7X7 stride 2로 변경되었다. 
또한 Conv3, 4, 5에서는 Filter의 수를 각각 384, 384, 256에서 512, 1024, 512로 늘려주었다. 결국 Filter의 크기는 작게하고 수는 늘려준 것이다. 


# VGGNet

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/1501685a-2d12-4eef-aa15-fe1628070431" height="50%" width="50%"></p>

기존의 AlexNet과 같은데서는 Conv와 MAX Pooling layer에서의 Filter를 계속 변경을 해주었는데, VGGNet에서는 오직 Conv layer에서는 3X3 stride 1 pad 1이고, Pooling layer에서는 2X2 stride 2만을 모든 레이어에 적용하였다. 
따라서 위 사진을 보면 몇개의 weight layer를 가지는 모델이 최적일지를 탐구한 것이다. 결과적으로는 D모델이 가장 최적이 모델이라는 결론이 나왔다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/3e488e8f-661d-4a28-baee-54b73b6dc400" height="70%" width="70%"></p>

최종적으로 D모델을 기준으로 분석하면 위 사진과 같다. image사이즈는 224, 112, 56... 으로 계속 줄어드는 것을 볼 수 있고, 반면 Filter의 수는 64, 128, 256, 512로 계속 늘어남을 알 수 있다.  
가운데 사용된 메모리를 보면 전체 합산하면 24M이 되고 float point 4바이트를 기준으로 하면 24M * 4 ~= 93MB / image가 된다.
이는 하나의 이미지 기준이며 forward 기분이고 backword까지 고려하면 이미지 한장을 처리하는데 약 200메가의 메모리를 사용하게된다.

메모리 사용 변화를 좀 더 자세히 보면 앞단 layer에서 주로 메모리를 사용하는 것을 볼 수 있고, 파라미터의 경우 뒤로 갈수록 파라미터의 수가 점점 늘어나면서 뒤쪽 FC에서 무려 1억개의 파라미터가 사용되는 것을 볼 수 있다. 이 때문에 FC layer를 사용하는 것이 효율적이지 않다는 판단이 나왔고, 최근에는 FC 대신 average Pooling사용에 대한 연구가 많이 되었다. 그래서 첫 FC layer를 보면 7*7*512*4096이 있을 때, 각 7X7을 average pooling을 해줘서 512개의 수를 가지는 단일 column으로 변환해준다. 실제 동작에서는 FC 만큼의 성능을 보이면서 파라미터 수는 크게 줄여줘서 매우 효율적으로 잘 동작하게 된다. 이 개념은 이후의 GoogLeNet에서도 사용된다. 


# GoogLeNet

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/3785ef24-6d28-4fe6-8294-c8d8a4eb7df5" height="100%" width="100%"></p>

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/c5bb9fb2-8ceb-4809-be66-15ff1154350d" height="50%" width="50%"></p>

GoogLeNet은 매우 복잡하다. 바로 위 사진을 Inception Module이라 하는데, GoogLeNet은 이런 Inception Module이 연속적으로 이어진 형태를 가지고 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/730f970c-9815-46c8-99fd-4ed8a0acb648" height="80%" width="80%"></p>

GoogLeNet의 구성을 잘 보면 앞서 언급한 avg pooling을 사용하는 것을 알 수 있다. avg pool을 보면 이전 layer에서 7x7x1024였던것이 avg pool을 거치고 1x1x1024가 됨을 알 수 있다. 
이렇게 단일 column으로 만들어 버림으로써 파라미터 수가 크게 줄었고 다 합쳐도 약 500만 정도밖에 되지 않는다. 

파라미터 수를 비교해보면 AlexNet : 60M, VGG : 138M, GoogLeNet : 5M 으로 크게 줄었다. 


# ResNet
ResNet은 이전 아키텍쳐들에 비해 ImageNet Classification 뿐만 아니라 Detection, Localization 그리고 COCO Detection, COCO Segmentation의 분야까지 모두 1등을 하였다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/ccc36021-50f9-4e2f-893d-9ad5f2d6db8b" height="60%" width="60%"></p>

이전 아키텍쳐 부터 layer의 수를 보면 매우 얕은 shallow부터 시작해서 천천히 증가하다가 ResNet에서 152개로 크게 증가함을 볼 수 있다. 
반면 Error rate는 점점 감소하게 된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/1c52a459-8fcb-472a-8e1f-3d5672188378" height="70%" width="70%"></p>

기본적으로는 layer가 많아질 수록 더 좋은 성능을 가져와야 한다. 하지만 문제들도 존재한다. 
ResNet이 말하는 바는 Alex나 VGG와 같은 것들은 layer가 많아짐에 따라 error가 오히려 더 커지므로 현재까지의 아키텍쳐들은 
최적화에 실패했다는 것이다. 
반면 ResNet같은 경우 layer가 많아질수록 정상적으로 error가 줄어들고 있다. 따라서 점점 많은 layer를 사용하기 위해서는 ResNet의 layer를 따라야 한다고 한다. 

ResNet의 놀라운 점은 layer의 수가 이전 아키텍쳐들에 비해 매우 급격하게 증가했다는 것이다. 
이에 따라 8개의 GPU로 2~3주 동안 오래 학습시켜야 하는대신 test 시에는 매우 빠른 속도를 보인다. 

### Residual block

H(x)를 기존의 네트워크라고 할 때, H(x)를 복잡한 함수에 근사시키는 것 보다 F(x) := H(x) - x일 때, H(x) = F(x) + x이고, F(x) + x를 근사시키는 것이 더 쉬울 것이라는 아이디어에서 출발하였다.
원래 Output에서 자기자신을 빼는 것이 F(x)의 정의이므로, 'Residual learning'이라는 이름을 갖게 된다. 
x가 F(x)를 통과하고 나서 다시 x를 더해주기 때문에 이를 Skip Connection이라고 부른다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/bc359a22-3e71-42d9-80fb-690311db6f5f" height="50%" width="50%"></p>

x : 입력값  
F(x) : CNN Layer -> ReLU -> CNN Layer 을 통과한 출력값   
H(x) : CNN Layer -> ReLU -> CNN Layer -> ReLU 를 통과한 출력값   

기존 신경망은 H(x)가 정답값 y에 정확히 매핑이 되는 함수를 찾는것을 목적으로 했다. 즉 신경망은 학습을 하면서 H(x) -y 의 값을 최소화시키면서 결국 H(x) = y가 되는 함수를 찾았다. 
위 그림에서 H(x)는 Identity를 매핑해주는 함수이기 때문에 H(x)-x를 최소화하면서 H(x) = x 가 되는 것을 목표로 한다. 

기존 신경망이 H(x) - x = 0을 만들려 했다면 ResNet은 H(x) - x = F(x) 로 두어 F(x)를 최소화 시키려고 한다. 
즉 F(x) = 0 이라는 목표를 두고 학습을 진행한다. 이렇게 학습을 진행하면 F(x) = 0이라는 목표값이 주어지기 때문에 학습이 더 쉬워진다. 결국 H(x) = F(x) + x 가 되는데 이때 입력값인 x를 사용하기 위해 쓰는 것이 Skip Connection이다. 즉 Skip Connection은 입력 값이 일정 층들을 건너뛰어 출력에 더할 수 있게 하는 역할을 한다. 

Plain과 비교했을 때 ResNet은 아래와 같이 Shortcut으로 이어진 것을 볼 수 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/bdf5b01f-7108-4faa-8001-0a913c4a5e41" height="30%" width="30%"></p>

#### Deeper bottleneck architecture
50층 이상의 깊은 모델에서는 Inception에서와 마찬가지로, 연산상의 이점을 위해 "bottleneck" layer (1x1 convolution)을 이용했다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/28ed5754-ecd3-4613-b21a-207672792cd0" height="50%" width="50%"></p>

기존의 Residual Block은 한 블록에 Convolution Layer(3X3) 2개가 있는 구조였다. Bottleneck 구조는 오른쪽 그림의 구조로 바꾸었는데 층이 하나 더 생겼지만 Convolution Layer(1X1) 2개를 사용하기 때문에 파라미터 수가 감소하여 연산량이 줄어들었다.
또한 Layer가 많아짐에 따라 Activation Function이 증가하여 더 많은 non-linearity가 들어갔다. 
즉 Input을 기존보다 다양하게 가공할 수 있게 되었다.

결론적으로 ResNet은 Skip Connection을 이용한 Shortcut과 Bottleneck 구조를 이용하여 더 깊게 층을 쌓을 수 있었다.

ResNet 초창기 논문에서는 110 Layer에서 가장 적은 에러가 나왔고, 1000개 이상의 layer가 쌓였을 때는 오버피팅이 일어났다고 한다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/dcfd57ba-d022-486a-94b1-60fd925af192" height="50%" width="50%"></p>

원 논문에서는 자신들이 어떤 이론을 바탕으로 결과를 도출했다기 보다는 경험적으로 residual block을 쓰니 결과가 좋게 나왔다고 말한다. 그럼에도 불구하고 ResNet이 왜 잘 동작하는지 설명하려는 많은 노력이 있었고, 아직까지도 많은 사람들이 연구하고 있었고, 그 중에서 소개할 한 가지는 바로 Residual Net이 만들어내는 모델을 이용하는 것은 Optimal depth에서의 모델을 사용하는 것과 비슷하다는 가설이다. 

우리는 쉽게 Optimal depth를 알 수가 없다. 20층이 Optimal인지, 30층이 optimal인지, 100층이 optimal인지 아무도 모른다. 
하지만, degradation problem은 야속하게도 우리는 알 수 없는 optimal depth를 넘어가면 바로 일어난다. 

ResNet은 엄청나게 깊은 네트워크를 만들어주고, Optimal depth에서의 값을 바로 Output으로 보내버릴 수 있다. 이게 가능한 이유는 바로 Skip connection 때문이다. ResNet은 Skip connection이 존재하기 때문에 Main path에서 Optimal depth이후의 Weight와 Bias가 전부 0에 수렴하도록 학습된다면 Optimal depth에서의 Output이 바로 Classification으로 넘어갈 수 있다. 즉 Optimal depth이후의 block은 모두 빈깡통이라는 것이다. 

예를들어 27층이 Optimal depth인데 ResNet 50에서 학습을 한다면, 28층부터 Classification 전까지의 weight와 bias를 전부 0으로 만들어버린다. 그러면 27층에서의 output이 바로 Classification에서 이용되고, 이는 Optimal depth의 네트워크를 그대로 사용하는것과 같다고 볼 수 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4ca311bb-081c-41ff-a4e3-66abfd5e4198" height="60%" width="60%"></p>

ResNet은 위와 같은 특성을 갖는데, Learning rate가 높은 이유는 batch normalization을 사용하기 때문에 높은 수치로 시작한 것이고 에러가 정체될 때마다 10으로 나눠준 것을 볼 수 있다. 또한 batch normalization을 사용했기 때문에 drop out 역시 사용하지 않았다.

최근의 트랜드는 더 작은 filter를 사용하려 하고 깊은 아키텍쳐로 구성하려고 한다. 
또한 Pooling과 FC layer를 사용하려 하지 않고 그냥 CONV layer만을 사용하려 한다. 



# Reference 
https://www.youtube.com/watch?v=rdTCxAM1I0I&list=PL1Kb3QTCLIVtyOuMgyVgT-OeW0PYXl3j5&index=6   
https://linkinpark213.com/2018/04/22/resnet/    
https://wikidocs.net/137252
