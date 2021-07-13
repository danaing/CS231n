# CS231n
# Lecture-1-Orientation
- Computer vision overview
- Historical context

## History of Computer Vision

(여기에 아이패드로 필기한 부분 적어넣기)


![image](https://user-images.githubusercontent.com/62828866/124767641-f420fb00-df72-11eb-9657-9cc21f40dd61.png)
object recognition이 어려우니까
object segmentation을 하자
그룹화하는 것이다.


graph theory algorithm을 가지고 버클리에서 한 segmentation이다.
그러다가 1999년에 머신러닝이 생겨나면서 발전하게 됐는데
그 중 가장 큰 기여를 한 기술은 2001년에  AdaBoost algorithm을 real -time face detection에 사용한 것이다.


21세기가 되면서 한가지가 바뀌었는데, 디지털카메라와 인터넷으로 사진의 퀄리티가 좋아져서 Computer vision을 연구하 더 좋은 데이터를 가질 수 있게 되었다!

Object Recognition의 발전은 측정할 수 있는 가장 influential한 benchmark dataset은 PASCAL Visual Object Challenge라는 것인데 2007년부터 2012년까지 꾸준히 향상되었다.


그 즈음에 프린스톤부터 스탠퍼드까지, 세상의 모든 것을 인식하고 싶다는 것과 데이터의 양이 너무 작아서 머신러닝처럼 그 데이터에 오버피팅된다는 문제를 해결하기 위해 ImageNet이라는 3년에 걸친 엄청난 프로젝트를 시작했다. 2009년에 Visual Recognition Challenge를 열었고 이로 인해 엄청난 발전이 있었음.

![image](https://user-images.githubusercontent.com/62828866/124773332-d4400600-df77-11eb-9906-d2621cac6e92.png)

![image](https://user-images.githubusercontent.com/62828866/125198455-e33af700-e29c-11eb-8e39-707901c72ef0.png)

2010년부터 시간이 갈수록 이미지 분류의 오차율(error rate)이 사람(human)을 이길만큼의 엄청난 발전이 있어왔다. 여기서 우리는 10%나 줄어든 2012년을 주목해야한다. 이때 CNN 모델이 당시 다른 모든 알고리즘 제치고 이기면서 엄청나게 큰 개선을 만들었다. 물론 지금은 더 퍼포먼스가 좋지만 그때의 발전에 대해서 주목할 필요가 있다. 이것이 바로 우리가 배울 강의의 focus이다. Convolutional Neural Network가 무엇인지, 그 Model은 무엇인지,  그 Principle은 무엇인지, 최근 발전은 어떤지 등등.

2012년에 CNN이 Computer Vision 분야에 큰 발전을 위해 보여준 tremendous capacity와 ability는 NLP와 Speech Recognition에도 같은 발전을 보여줬다.


---------------------

## Overview of CS231n

기본적으로 **image classification**을 배우는 코스이다. 기본적(general)한 것이라서 이것은 useful 하고 다른 곳에도 apply될 수 있다.

Visual Recognition의 다른 종류도 있다.
- object detection
- image captioning
- Action classification
- ...

위 분야는 약간 problem setting이 다르지만 역시나 기본적으로 CNN을 사용한다.


![image](https://user-images.githubusercontent.com/62828866/125194928-5c7f1d80-e28e-11eb-99b4-1969b5143dc2.png)

- 2011년 Lin et al.이 제시한 논문은 여전히 계층적이고 여러 레이어로 구성되어 있다. 마지막 층은 Linear SVM이다. 여전히 이미지의 가장자리를 감지하고(detecting edges), 불변(invariance)의 법칙을 가지고 있다.

- 2012년에 토론토에 있는 Jeff Hinton의 그룹이 Alex Krizhevsky와 그의 박사생과 함께  7 layer CNN 을 만들어서 큰 발전을 가져왔다. AlexNet으로 알려 있음. 2012년에 다 이김. 7~8개의 레이어이다. 그 이후에는 모든 승자가 cNN이였고 점점 깊어졌다.

- 2014년에는 Google의 GoogleNet과 Oxford대학교의 VGG Net 19 레이어

- 2015년에는 Microsoft Research Asia에서 만든 152개 레이어(really crazy)로 된 Residual Network

- 그 이후에는 200개 이상의 레이어를 사용하거나 좋은 GPU를 사용하면 성능을 올릴 수 있지만, 그것은 나중에 배울 것이다.

그러나 가장 중요한 것은, CNN은 2012년에 정말 획기적(breakthrogh moment)
이란 것이고, 이것이 어떻게 작동하는 지를 이번 코스를 통해 배울 것이다.


![image](https://user-images.githubusercontent.com/62828866/125196445-f3e76f00-e294-11eb-98d5-6a6c5046f0f2.png)

놀라운 사실은 2012년에 ImageNet에서 승리한 CNN은 그때 발명된 것이 아니다. 아주 전에 이미 고안된 것이다.

Jan LeCun이 Bell Labs에 있던 사람과 협력해서 1998년에 숫자 인식을 위해 CNN을 고안했다. 실제로 2012년에 제안된 AlexNet과 비슷한 아키텍쳐이다.

그러면 이것이 왜 2012년 이후에 인기를 얻었는가?라고 질문할 수 있다.

거기에는 중요한 혁신(key innovations)이 있다.
1. **Computation의 발전**: 무어의 법칙(Moore's law), transistor의 개수, GPU 덕분에 더 큰 아키텍쳐와 모델을 계산할 수 있게 되었다. 이것이 가장 중요한 이유
1. **Data의 퀄리티**: PASCAL, IMAGE 높은 화질의 좋은 많은 데이터셋이 생겨서 train할 수 있게 되었다.

이렇듯 CNN은 근래에 폭발적으로 발전해서 fancy한 알고리즘처럼 보이지만 오랫동안 존재했던 것입니다.

아직 Computer Vision에는 많은 open challenge들이 남아있다. Semantic segmentation, Activity recognition, Visual Genome, 그리고 이미지에 대한 깊은 이해(deep understanding)를 하기 위해서는 아직 연구할 것이 많이 남아있다.

![image](https://user-images.githubusercontent.com/62828866/125198545-3d3bbc80-e29d-11eb-9539-0c208f6a0914.png)

이 수업에 대한 우리의 철학(Our Philohophy about this class)은 진짜 이 모든 알고리즘의 깊은 메커니즘에 대해 이해해야한다는 것이다.
