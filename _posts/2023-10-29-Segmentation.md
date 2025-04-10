---
title:  "[CS231n] 12.Segmentation"
excerpt: "CS231n Lecture Note"

categories:
  - Computer Vision
tags:
  - CS231n
toc: true
toc_sticky: true
last_modified_at: 2023-10-29T08:06:00-05:00
---

# Segmentation

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/96044369-69ae-4bb0-a3ef-b22e2a79005b" height="80%" width="80%"></p>

앞서 Object Detection까지 살펴보았고 이제 Segmentation에 대해 알아볼 것이다. Segmentation은 크게 Semantic Segmentation과 Instance Segmentation으로 나뉜다. 


## Semantic Segmentation

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/3ba1ee55-bae7-4da7-be62-fdf052eefb0c" height="70%" width="70%"></p>

Semantic Segmentation은 딥러닝이 발달하기 전에 나온 방식으로 기본적으로 classification은 one label per image로 이미지 기준으로 label해주는 것이었다면 
Semantic Segmentation은 one label per pixel 이다. 그래서 픽셀 단위로 label해주게 되고 모든 픽셀이 하나의 class를 갖게 되는 것이다. 
그러다 보니 instance들을 구분하지 못하게 되는데 위에서 소 사진을 보면 소가 4마리가 있는데 이를 각각 인식하지 못하고 하나의 cow로 인식하는 것을 볼 수 있다. 
그리고 class의 수는 아래 여러 색들처럼 정해진 수의 class를 갖게 된다. 또한 background class라는 것을 뒤서 특정 class에 해당되지 않는 것들을 다 background class로 구분한다. 

Semantic Segmentation의 기본적인 pipline은 다음과 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/05c590f5-50c8-47fc-bfd5-a73e1aa99b7d" height="70%" width="70%"></p>

이미지가 있으면 특정 부분을 patch(추출)하고 이를 CNN에 돌린다. 이렇게 하고 가운데 있는 픽셀이 어떤 class인지 classify하는 것이다. 그리고 이 과정으로 모든 각각의 patch에 대해 반복해 준다. 
하지만 이는 매우 많은 patch가 나오므로 비용이 매우 큰 방식이다. 

그래서 이런 비용이 많이드는 연산을 피하기 위해 나온 트릭이 fully connected layer를 사용하지 않고, 전체 layer를 다 fully convolutional하게 구성하는 것이다.  

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/365b705f-1ecd-40b8-aabe-bad330553429" height="70%" width="70%"></p>

fully convolutional network에서는 이미지 자체를 CNN에 넣어서 모든 픽셀에 대한 결과를 한 번에 구해내는데 문제는 CNN의 pooling이나 stride convolution 등이 있기 때문에 down sampling이 일반적으로 발생하고 이로 인해 output image가 위 사진과 같이 매우 작아지게 된다. 

### Multi-Scale

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/12ecd388-33b3-40d2-bed0-520c2420a668" height="70%" width="70%"></p>

Semantic Segmentation의 확장된 버전 중 하나는 Multi-Scale이다. 이는 여러개의 Scale을 이용하는 것으로 위 사진과 같이 다양한 Scale로 이미지를 resize해주고 
이렇게 scaling된 이미지를 각각 별도의 CNN으로 돌려준다. 이렇게 하면 결과로 나오는 feature map은 각기 크기가 다르게 된다. 그래서 그 다음 단계에서 이 결과들을 원본과 같은 크기로 upscale해주게 되고 이것들을 concatenate 해준다.

그리고 이와는 별도로 offline porcessing해주는 과정이 필요한데 이는 bottom-up segmentation이라고 해서 superpixel과 tree 두가지 방법을 이용할 수 있다. 
superpixel은 근접한 픽셀들을 보고 큰 변화가 없으면 연관된 영역으로 묶어서 영역을 구분해나가는 것이고 tree는 어떤 segmentation들이 같이 merge되어 나가야 할지를 결정해주는 tree이다. 
이러한 별도의 bottom-up segmentation은 위에서의 CNN에서는 줄 수 없는 큰 맥락에서의 추가적인 정보를 제공하는 역할을 한다.  
이렇게 해서 최종적으로 위의 과정과 아래 과정을 묶어주고 결과를 낸다. 


### Refinement 
Semantic Segmentation의 두 번째 확장 버전은 반복적으로 refine작업을 해나가는 Refinement이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/d8a626c3-2ec6-408e-b7fd-8f62c57e4dc1" height="70%" width="70%"></p>

위 사진처럼 input 이미지를 RGB채널이 분리된 상태로 CNN을 적용해줘서 이미지의 모든 label을 얻어내는데, 그 결과는 원본보다 downsampling된 버전이 나온다.
그리고 이러한 과정을 2번 더 해준다. 이렇게 3번의 CNN을 적용하는 동안 CNN들이 파라미터인 weight들을 공유하게 되고 이는 마치 RNN처럼 동작하게 된다.

이러한 반복과정은 많아지면 많아질수록 결과가 더 좋아진다고 한다.  


### Upsampling 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/1e3fec29-2bcd-46dd-955e-15b1d41681d4" height="60%" width="60%"></p>

세 번째 방식은 Upsampling이다. 이는 매우 유명한 방법으로 앞 두가지 방법과의 공통점은 input 이미지를 받아서 CNN을 돌려서 feature map을 추출하는 것인데, 
결정적인 차이점은 이미지의 feature map 점점 작아지는데 그 작아진 feature map을 다시 크게 복원하는 upsampling방법에 있어서 기존의 별도의 작업을 이용하지 않고 
upsampling 작업까지도 이 네트워크의 일부분으로 넣었다. 그래서 위 사진처럼 학습이 가능한 upsampling layer를 추가하였다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/1aec3bf3-e9ee-4d45-9f58-27a0ca9ff914" height="70%" width="70%"></p>

이러한 CNN은 두 가지 특징을 갖는데, 첫 번째는 Upsampling이다. 학습가능한 Upsampling을 통해서 32배의 크기 확장을 시키는 것이다.  
그렇게 하므로서 복원을 하는 것이다. 
또 하나의 특징은 skip connection이다. Convolution 과정의 앞 단계 일수록 receptive field는 더 작다. 그렇기 때문에 원본이미지의 세밀한 구조 파악에 도움을 주는 것은 앞단 layer가 된다.       
그래서 여기서의 skip connection은 끝까지 down sampling된 것을 Upsampling하는 것 뿐만 아니라 추가적으로 위 사진의 pool3이나 pool4와 같은 중간 단계의 pooling 단계에서 또 다른 feature map을 추출하는 것이다.
그래서 이렇게 나온 결과를 다시 각각 Upsampling해주고 위 사진에서는 3가지인 결과를 전체 결합해줘서 최종적인 결과를 낸다.   
왼쪽 아래 결과 사진은 오른쪽으로 갈수록 skip connection 결합해준 각각의 결과들이 많아지게 되고 결과도 더 좋아짐을 보인다. 

#### Deconvolution 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/21eba4f7-f39a-4fe6-b8f7-966d543757d4" height="70%" width="70%"></p>

그래서 이렇게 학습 가능한 Upsampling을 Deconvolution이라고 하는데 기존 conv와는 다르게 input에 filter를 적용해서 각각의 element간의 dot product를 하지 않고 
반대로 filter를 output에 copy하여서 input의 단일 weight이 output의 weight값들을 부여해준다. stride 역시 반대로 input에서는 1씩 이동하면 output에서는 stride 값 만큼 이동해서 weight들 제공해준다. 
그리고 이 과정에서 output에서 겹치는 부분은 sum을 해준다. 

사실 이 Deconvolution이라는 용어는 부적절하고, 적절한 용어는 convolution transpose, backward strided convolution 등이 있다고 한다.  


## Instance Segmentation

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/2a0a5b75-e2de-4844-8177-17bc0b63eb37" height="70%" width="70%"></p>

Instance Segmentation은 우선 Instance들을 detect한다. 예를 들면 사람 하나하나를 다 detect하고 각각의 Instance내의 픽셀들을 label해준다. 
사진을 보면 Instance를 먼저 detect하고 각각을 person으로 label해줘서 Instance간의 구분을 해주는 것을 볼 수 있다. 
그래서 Semantic과의 근본적인 차이는 Instance를 구분하는냐 못하느냐가 된다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/beaa7b34-dfcc-40a2-82fd-befac7e83464" height="70%" width="70%"></p>

Instance Segmentation은 사실 R-CNN과 매우 유사한데 input 이미지를 받아서 마치 R-CNN에서 region proposal을 해준 것 처럼 오프라인으로 Segment proposal을 해준다. 
이를 통해 각각의 Segment proposal된 것에 대해 feature를 뽑아내고 이를 CNN에 돌리고 이를 Box CNN이라 한다. 이러한 Box CNN에서는 bounding box를 결과로 얻게 된다.

또 하나의 과정은 Region CNN인데 Box CNN에서는 cropping한 이미지를 그대로 돌리는 반면 region CNN에서는 mean color를 이용해서 배경을 제거하고 전경만 CNN으로 돌린다. 
이렇게 해서 Box CNN과 region CNN의 결과를 결합하고 해당 region에 대해 classification을 수행한다. 그리고 마지막으로 해당 region을 다듬어준다.   

### Hypercolumns 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/864a3a1e-5347-4536-93e4-09687623a1dd" height="70%" width="70%"></p>

Instance Segmentation도 Semantic처럼 논문이 나오고 발전된 버전이 나왔는데 첫 번째는 Hypercolumns이다.  
Hypercolumns은 multi scale과 매우 유사한데, 원본 이미지를 cropping해주고 alexNet으로 돌려준다. 그런데 여기서 convolution을 진행하는 과정 사이사이에 upsampling을 진행해준다. 
이렇게 upsampling해준 것들을 결합해서 결과를 내는 것이다.   


### Cascades

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/592a3a5d-2c29-4a71-b2b2-3560cd0be6cf" height="70%" width="70%"></p>

두 번째 버전은 Cascades이다. Cascades은 R-CNN의 진화 버전인 faster R-CNN과 매우 유사하다.   
방식은 faster R-CNN과 마찬가지로 이미지자체를 Convolution 시켜서 거대한 feature map을 만들고 faster R-CNN의 RPN을 그대로 이용해서 Box instance를 뽑아낸다. 
이때 생성된 box들은 다 크기가 다를 것인데 이것들의 size를 다 동일하게 맞춰준 다음에 이들을 FC에 넣어준다. 그리고 이들을 figure/ground logistic regression을 이용해서 mask instance를 생성해준다.
그리고 다시 한번 mask 과정을 해줘서 배경을 다 날려버리고 전경만 남게 한 뒤 이를 다시 FC에 넣어줘서 object를 classification하게 된다.  

# Reference 
https://www.youtube.com/watch?v=Q9bNbl5FiD8&list=PL1Kb3QTCLIVtyOuMgyVgT-OeW0PYXl3j5&index=11  
