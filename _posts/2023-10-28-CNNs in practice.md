---
title:  "[CS231n] 11.CNNs in practice"
excerpt: "CS231n Lecture Note"

categories:
  - Computer Vision
tags:
  - CS231n
toc: true
toc_sticky: true
last_modified_at: 2023-10-28T08:06:00-05:00
---

# Data Augmentation 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/805152c9-90be-4060-97f0-9745c4be4530" height="80%" width="80%"></p>

일반적인 CNN은 이미지와 label을 CNN에 feed해주고 loss를 구해서 최적화 하는 방법을 사용한다. Data Augmentation은 여기에 위 사진과 같이 input image를 변형하는 과정이 하나 추가된다. 

그래서 Data Augmentation은 label에는 변화가 없지만 픽셀에는 변화를 주고 이런 변경된 데이터로 학습을 하게 된다. 
간단한 예시 몇 가지로 아래와 같은 것들이 있다. 

## 1. Horizontal flips

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/6a340fb5-5188-4100-b8e7-fd62123be2d0" height="50%" width="50%"></p>

## 2. Random Crops/Scales

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/294755a9-f59e-4b8c-841b-8a49d440b0ac" height="20%" width="20%"></p>

Random Crops/Scales은 이미지를 랜덤하게 자르거나 scale을 다양하게 해서 학습시키는 것이다.

**ResNet**  
1. Pick random L in range [256, 480]
2. Resize training image, short side = L
3. Sample random 224 X 224 path

예를 들어 ResNet에서는 이미지를 [256, 480] 사이의 L을 랜덤하게 선택해주고 training image를 resize해준다. 여기서 짧은 부분이 L이 되도록 하고 이후 랜덤하게 224 X 224 크기를 갖는 patch를 샘플링하여 추출한다. 

이러한 Augmentation을 이용하면 training 시에 이미지 전체가 아니라 crop(부분 부분)에 대한 학습이 이루어지기 때문에 test 시에도 이미지 전체가 아니라 정해진 수의 crop을 이용해서 test를 진행하게 된다. 

**Test Example**  
1. 테스트할 이미지를 5개의 크기로 resize해준다. (224, 256, 384, 480, 640)
2. 각각의 사이즈에 대해 224 X 224의 크기를 갖는 10개의 crop을 만들어 준다. (코너 부분의 crop 4개 + 중심 부분의 crop 1개 = 5개 -> 이를 Horizontal flips까지 해줘서 총 10개. 각각의 size에 대해 10개씩 이므로 총 50개.)
3. 그리고 이 50개에 대해 평균을 구해준다. 


## 3. Color jitter

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/cc30b487-1e55-4065-a642-038c14c01ce5" height="80%" width="80%"></p>


Color jitter의 가장 간단한 방법은 contrast를 jittering 해주는 것이고 이보다는 복잡하지만 많이 사용되는 방법은 이미지 각각의 RGB 채널에 대해 
PCA(주성분 분석)을 해준다. 주성분 분석을 해주는 이유는 이미지의 주성분을 뽑아냄으로써 이미지의 핵심을 잃지 않으면서 이미지의 갯수를 늘려줄 수 있기 때문이다.
주성분 분석을 해준면 각각의 채널에 대해서 principal component direction을 얻게 된다. 이는 컬러가 변화해나가는 방향성을 파악하는 것이고 이러한 principal component direction을 따라 color의 offset을 샘플링을 해주고
이 offset을 이미지 모든 픽셀에 더해준다. 

위와 같은 방법 외에도 translation, rotation, stretching, lens ditortions(랜즈 왜곡 효과) 등 다양한 방법이 존재한다. 


<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/bc663151-21a8-4775-96d4-1056da8bd810" height="80%" width="80%"></p>

그래서 이러한 Augmentation을 크게 보면 training과정은 랜덤한 노이즈를 더해주는 과정이 되고 test과정은 이러한 노이즈를 평균화 하는 과정이라고 생각할 수 있다. 
이렇게 보면 큰 의미로 dropout이나 drop connect들도 Data Augmentation의 일부라고 생각할 수 있다. 
마찬가지 맥락으로 batch normalization, model ensembles 등도 이와 유사한 효과를 가지고 있다. 

정리하면 Data Augmentation은 구현이 매우 간단하기 때문에 사용하면 좋고, 특히 data set의 크기가 작을 때 유용할 것이다.  


# Transfer Learning 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/84039ac8-8c18-4dc0-b323-48e06ece0df1" height="80%" width="80%"></p>

CNN으로 Transfer Learning하는 과정을 살펴보면 우선 모델을 imageNet에 training시킨다. imageNet에 training시킨다는 것은 우리가 직접 처음부터 학습시킬 수도 있고, 미리 학습된 모델(pre-trained model)을 가져올 수도 있다.
dataset이 충분한 경우는 직접 학습시켜도 된다. 

이때 만약 우리의 dataset이 2번째 경우 처럼 너무 작은 경우에는 FC-1000과 Softmax만을 남겨놓고 나머지 layer에 대해 다 Freezing을 해준다. 즉 Freezing 해준 layer의 파라미터들은 변하지 않게 되고 이렇게 Freezing 해준 부분을 feature extractor처럼 생각할 수 있다. 그래서 우리의 dataset은 FC-1000, Softmax 이 부분에 대해서만 학습을 시키게 된다.  

만약 3번째 경우처럼 dataset이 너무 많지도 않고 너무 적지도 않은 경우에는 finetuning을 해주게 된다. 
사진에서 보이는 것처럼 가지고 있는 data의 양에 따라 Freeze하는 영역을 조율하며 학습을 시키게 된다. 
finetuning에서의 한 가지 팁이 있는데 finetuning하는 layer의 top layer(빨간색 박스의 연두색 정도의 부분)는 learning rate을 원래의 rate의 1/10정도를 사용하고 그 위(빨간색 박스의 주황색 정도의 부분)는 1/100, 그리고 그보다 위는 Freeze 부분이기 때문에 rate가 0 이렇게 설정한다고 한다.   

만약 Transfer Learning을 할 때 pre-trained된 모델이 학습한 class와 유사한 class들을 분류해야 한다면 2 번의 경우 처럼 끝 부분만 학습시켜도 성능이 좋고 만약 전혀 관련이 없는 데이터를 분류해야 한다면 Freeze하는 부분을 줄이고 학습시키는 layer를 늘려야 한다고 한다. 
그런데 의문점은 이렇게 학습시키는 layer를 늘리면서 전이학습을 진행한게 아에 처음부터 직접 학습시키는 것 보다 더 성능이 좋다고 하는데, 그 이유는 앞 layer의 filter를 보면 edge, color 등의 low level feature를 인지하고 뒷 layer로 갈 수록 점점 상위 레벨의 추상적인 것들을 인식하기 때문에 low level feature를 미리 학습시켜 놓는다는 것은 그 어떤 이미지를 분석하더라도 도움이 된다는 것이다. 그렇기 때문에 전이 학습이 더 좋은 성능을 낼 수 있는 것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/6d6dfe89-1055-4375-8b3f-37c8d4cd1e1c" height="80%" width="80%"></p>

그래서 가지고 있는 data의 수와 pre-trained된 모델이 학습한 데이터와 자기가 분류하고 싶은 데이터 set과의 유사성에 따른 관계는 위 표로 정리할 수 있다. 

결론적으로 CNN에서의 Transfer learning은 거의 항상 사용된다고 보면 되고 object detection과 같은 Faster R-CNN에서도 CNN에서는 전이 학습을 사용을 하고 Image Captioning의 경우도 CNN부분은 물론 RNN부분에서도 word vector에 대한 전이 학습을 사용한다고 한다. (word2vec pre-training)

이러한 pre-trained model은 아래 사이트에서 찾아볼 수 있다.   
https://github.com/BVLC/caffe/blob/master/docs/model_zoo.md


# All About Convolutions
## Part1 : How to Stack them 
이제는 Convolutions에 대해 좀 더 자세히 살펴볼 것인데, 첫 파트는 어떻게 stacking을 할 것인가에 대해 알아볼 것이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/6e3379bf-117e-4ea4-9a6a-44f532aba910" height="80%" width="80%"></p>

2개의 3X3 (stride 1) layer를 쌓는다고 가정을 하면 위 사진과 같이 될 것이다. 이렇게 되면 conv layer에서 하나의 뉴런은 그 전 단계의 activation map에서 3X3의 지역을 보게될 것이다. 위 사진에서의 Second Conv layer는 Input layer의 어느 정도의 크기를 보게될 것인가에 대한 대답은 5X5이다. 이유는 First Conv에서 3X3으로 9개를 보고 각각의 뉴런이 또 Input에서 3X3을 보게 된다. stride는 1 이기 때문에 결국엔 input layer의 모든 영역을 보게 되어 5X5가 되는 것이다.   

그렇다면 이번엔 3X3 conv layer를 3개를 쌓는다고 했을 때 3번째 layer는 input layer에서 어느 정도의 크기를 보게 될까?
정답은 7X7이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/2fdc7143-b8b6-4c80-acd8-b29254436ef7" height="70%" width="70%"></p>

이유는 앞서 설명한 것과 같고 여기서 중요한 점은 3X3 conv layer 3개를 쌓은것이 결과적으로 단일 7X7 layer와 동일한 representational power를 갖는다는 것이다. 물론 직관적으로는 이렇고 내부에는 여러 차이가 있을 수 있다. 

앞서 공부한 내용을 생각해보면 input이 32x32x3이고, 10개의 5x5 filter, stride 1, padding 2라고 할 때 파라미터의 개수를 계산해보면 
`(5x5x3+1)x10 = 760`이 된다. 이는 5x5 filter니까 5x5를 해주고 input의 depth가 3이니까 3을 곱해주고, bias 1을 추가해준 것을 계산하고 이러한 filter가 10개가 있으니까 10을 곱해서 총 760이 된다.    

그래서 만약 input이 `H x W x C` 라고 하고 `C`개의 filter로 conv를 할 때 하나의 7X7 filter와 3개의 3X3 filter의 파라미터를 비교를 해보면 아래와 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/40d18ecb-5053-4cad-bef6-3e3a9c97a1cb" height="80%" width="80%"></p>

3개의 3x3의 경우 말그대로 conv layer가 3개가 있으니까 filter 수 만큼 곱하고 3을 또 곱한 것이다.
결론적으로 파라미터 수를 비교해보면 3개의 3x3이 더 적은 것을 볼 수 있고 3번의 conv layer를 거치다 보니 그 안에 relu를 계속 거치게 되고 이로 인해 nonlinearity도 더 강화된다. 이를 통해 작은 filter를 쌓는 것의 장점을 알 수 있다. 

여기에 더해 얼마나 곱셈 연산이 발생하는가도 비교해보면 input이 H x W x C이니까 H x W x C를 계산한 값에 7x7 filter의 경우 7x7xC가 된다. 그래서 이를 모두 곱하면 $49HWC^2$ 이 되고 3x3filter도 마찬가지로 계산하면 $27HWC^2$ 가 된다. 그래서 이렇게 곱셈 연산을 보더라도 3x3 filter의 경우가 연산이 더 적음을 알 수 있다.  

그렇다면 더 작은 filter가 더 좋은 성능을 낸다면 1x1을 쓰면 안되냐라고 생각할 수 있다. 이에 대한 예시는 아래 사진과 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/e2ac1a69-2e8e-4009-80da-dc2dc18dd8b0" height="80%" width="80%"></p>

위 사진과 같이 HxWxC 차원에 Conv 1x1, filter를 C/2로 돌리는데 이렇게 1x1로 돌리는 것을 bottleneck이라고 한다. 그리고 다음 단에서는 3x3 conv를 이용하고, 그 다음 단계에서는 다시 1x1 conv를 이용하는데 C개의 filter를 이용해서 dimension을 다시 복원해준다. 

이런 식으로 1x1 conv를 활용하는 것을 Network in Network 또는 bottleneck이라고 표현하며 이러한 구조는 유용성이 많이 입증되었다.

그래서 이런 bottleneck을 활용하는 것과 그냥 3x3 conv, C filters를 활용한 결과를 비교하면 결과는 HxWxC로 동일한데 사용되는 파라미터의 수를 계산해보니 bottleneck에서는 $3.25C^2$, 3x3 conv에서는 $9C^2$가 되었다. bottleneck의 계산은 이전과 마찬가지로 첫 conv에서는 1x1xC(depth)를 계산하고 filter의 수 C/2를 곱해서 $\frac{C^2}{2}$가 되고, 다음 layer는 $\frac{9C^2}{4}$ 마지막 layer는 $\frac{C^2}{2}$가 되어 다 더하면 $3.25C^2$가 된다. 이렇게 마찬가지로 nonlinearity도 좋아지고 파라미터도 적어지고 연산도 적어진다.      
<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/b05b60c2-218f-4d80-bec4-14642513e507" height="60%" width="60%"></p>

위와 같은 방식으로도 연산을 줄일 수 있는데, 여기에 1x1를 더 섞어주면 연산이 더 줄어든다. 이런 방식을 최근의 네트워크는 많이 사용하고 있고, 대표적으로 아래 GoogLeNet의 예시이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/18e874c4-ec62-48b5-a0e8-fa1a133e334e" height="70%" width="70%"></p>

사진을 보면 1x1의 bottleneck을 수 많은 곳에서 활용하고 있는 것을 볼 수 있다.  

그래서 내용을 총 정리해보면 아래와 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/b86f3a0f-ccee-4567-a8d6-9df3bf1d701d" height="60%" width="60%"></p>

## Part2 : How to compute them 

### im2col
지금까지는 conv를 어떻게 stacking 할지에 대해 알아보았고 이제는 어떻게 연산할 것인지를 볼 것이다. 
conv를 구현하는 방식이 여러가지 있는데 이 중에 하나가 im2col(image to column)이라는 것이다.
행렬곱은 원래 매우 빠른 연산이고 대부분의 플랫폼에서 이러한 행렬곱을 잘 구현해 놓았다. 
그래서 convolution을 행렬곱의 형태로 표현하기 위해 im2col이 나왔다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/0d7f7a2e-3817-4d4c-8963-082e8c08c837" height="70%" width="70%"></p>

구현 방식은 feature map이 HxWxC이고 conv는 KxKxC의 D개의 filter라고 할 때 첫 단계에서 이 KxKxC의 receptive field를 $K^2C$의 원소를 갖는 column vector로 reshape해준다. 그리고 이러한 receptive field의 개수를 N이라 한다면 $K^2C$의 column vector를 N번 만큼 반복한 것이 되는데 여기서 한 가지 문제점이 있다면 receptive field의 원소들이 중복되어 메모리를 낭비한다는 것이다. 그러나 이는 큰 문제는 아니라 일반적으로는 효율적이고 다음 단계에서는 이번엔 filter를 low vector로 reshape해준다. 그래서 KxKxC를 $K^2C$로 하고 filter의 수가 D이기 때문에 D를 곱해준다. 그래서 이렇게 만들어진 두 행렬을 행렬곱을 해서 결과를 구한다. 위 사진에서는 행렬곱의 순서가 잘못되었는데 사실은 거꾸로 되어야 한다.($D \times (K^2C)$가 앞으로 가야한다.)         

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/d0e0360f-aeb0-4e47-a9a3-5488d99641f1" height="70%" width="70%"></p>

그래서 이러한 im2col은 실제 많이 사용되고 위는 순전파 시 im2col를 사용한 예시 코드이다.  

### FFT

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/ef1774ad-8b87-49fb-8291-ca51f9e9ddc3" height="70%" width="70%"></p>

이제는 FFT(Fast Fourier Transform)에 대해 알아볼 것이다. 이는 signal processing에서 convolution theorem이라는 것이 있는데, 
convolution theorem에 의하면 signal f와 g가 있을 때, 이 둘을 convolution을 한 것이 elementwise한 곱을 한것과 동일하다는 것이다.
그래서 위 사진과 같은 식으로 표현할 수 있고 결국 Fast Fourier Transform라는 것은 푸리에 변환과 그 역행렬을 매우 빠르게 계산하는 알고리즘이다.   

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/5b70afea-0785-43b4-b07f-72e1a7cf5ecc" height="60%" width="60%"></p>

그래서 이를 이용해서 weight를 구하고 activation map에 대해 구하고 이들을 elementwise product를 해준다. 그리고 결과에 역행렬을 취하는 식으로 convolution을 진행하게 된다.   

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/ed54242e-bce3-42a7-99cf-9caf4ba09db3" height="70%" width="70%"></p>

결과를 보면 초록색 부분이 speed up이 된 부분인데 filter가 큰 경우 효과가 많이 있는데 작은 filter를 사용할 때는 효과가 별로 없다는 것이 나타났다. 

### Fast Algorithms 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/7b872cd6-627c-49fd-9dfd-526f8380b1b5" height="70%" width="70%"></p>

또 따른 convolution 구현의 예로는 Fast Algorithms이라는 것이 있는데, 이는 일반적으로 NxN의 행렬 연산을 하면 $O(N^3)$의 결과가 나오는데, Strassen's Algorithm을 이용하면 시간복잡도를 줄일 수 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/14cbb700-d39e-4df2-8cf7-b032a9d0e677" height="60%" width="60%"></p>

이러한 효과를 convolution에 적용하여 연산한 것이 위 사진이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/74b0d530-efe7-43a0-bd00-ba927a3570c2" height="70%" width="70%"></p>

그래서 이를 VGG Net에 테스트해 보았는데 상당한 speed 향상을 가져온것이 확인되었다. 

단점은 conv의 사이즈가 다르면 각각의 사이즈에 맞춰서 알고리즘을 최적화 해줘야 한다는 것이다. 
그럼에도 불구하고 속도가 매우 빠르기 때문에 많이 사용될 것으로 보인다. 


# Implementation Details
이제는 실제 convolutional net을 어떻게 구현할 것인지를 살펴볼 것이다. 
GPU는 느린 코어가 매우 많이 있어서 parallel한 연산에 매우 강하고 이렇다 보니 딥러닝에서 많이 쓰이게 된다. 
GPU는 또한 프로그래밍이 가능한데 예를들어 NVIDIA의 CUDA같은 경우 GPU상에서 바로 실행이 가능한 C 코드를 작성할 수 있다. 
그리고 여기서 제공되는 API로는 행렬연산에 최적화된 cuBLAS, 딥러닝에 최적화된 cuDNN 등이 있다. 

하지만 현실적으로는 GPU를 아무리 많이 사용한다고 해도 학습에는 오랜시간이 걸리는데 VGG같은 경우 메모리를 매우 많이 사용해서 여러개의 GPU를 사용하고 예를 들어 mini batch가 128개로 구성된 경우 이 128개를 각가 32개씩 해서 4개의 GPU로 연산하고 각각의 weight들을 합쳐서 결과를 update할 수 있다.     

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/0af8ea24-d7db-4bd1-8504-18ff71fb7294" height="80%" width="80%"></p>

여러개의 GPU를 사용하려면 좀 더 복잡해지는데, Google이 사용한 방식은 원래 일반적으로는 Synchronous한 처리를 하는데 이는 왼쪽 사진처럼 각각의 model이 있는 부분(GPU)에 mini batch를 할당하고 각각의 GPU가 계산한 값들을 통합해 준다. 이 과정에서 각각의 GPU들을 synchronize해주는 과정이 매우 비효율 적이다. 그래서 구글이 개발한 오른쪽의 경우는 각 GPU model이 학습을 하면 한 번에 동기화하여 update하지 않고 각각의 모델이 gradient를 계산할 때마다 그때그때 개별적으로 update를 진행하게 된다.         

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/b817db42-5eb9-4b97-9195-80ebb9d72622" height="40%" width="40%"></p>

딥러닝을 수행하는데 있어서 여러 bottleneck들이 일어나게 되는데 우선 CPU와 GPU의 통신이 불가피하게 일어날 수밖에 업고 이것이 하나의 큰 bottleneck이 될 수 있다. 매 번 순전파와 역전파를 해줄 때마다 data를 GPU에 복사해줬다가 이를 다시 CPU로 복사해오는 작업에서 큰 부하가 걸린다. 그래서 CPU에서는 미리 데이터를 fetch해오고 GPU에서는 순전파와 역전파의 모든 과정이 가능해야 이러한 bottleneck을 없앨 수 있다.   






# Reference 
https://www.youtube.com/watch?v=8kzgwfNSDfk&list=PL1Kb3QTCLIVtyOuMgyVgT-OeW0PYXl3j5&index=10&t=222s
