### Project Overview

Adaptive Learning Platform for K-12 Math


### Structure
Scalability를 위해 django app 구조를 아래와 같이 변경함(241107)
```
p4ds: main app
user: 사용자 관리
practice: 문제풀이 관리
analytics: 분석 기능
```
문제 추천 등은 util로 따로 정리해야 함

### 중요사항
* Django에서는 User 관리를 위한 별도의 체계를 제공하고 있음. AbstractUser라는 클래스를 상속받아서 User 클래스를 만들어서 사용해야 편리하게 사용가능함.
* url 관리를 위해서 사용하는 방법으로, 각 app의 내부에 `urls.py` 파일을 만들고, config의 `urls.py`에서 이를 refer하게 함으로써 모듈러한 개발이 가능함.
* 각 app의 내부에 위치한 `urls.py`에서 정의된 이름을 바탕으로 django가 해당 주소를 불러올 수 있게 됨.
* page template을 위한 base layout 파일을 만들고(header, footer 등을 포함시키는 기본 틀), 이를 extend하여 각 템플릿을 제작함.

* 기본 css 대신 sass를 사용하여 편의성을 높임(`pip install sass`, `pip install django-sass-processor` 필요)

### 필수 라이브러리 설치
```
pip install -r requirements.txt
```
### 비동기 동작을 위한 사항
1. rabbitMQ가 동작할 docker 생성
```
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:management
```
2. docker 내에서 myuser 생성 및 권한 부여
```
docker exec -it rabbitmq bash

# docker 내에서 입력
rabbitmqctl add_user myuser mypassword
rabbitmqctl set_user_tags myuser administrator
rabbitmqctl set_permissions -p / myuser ".*" ".*" ".*"
```
3. runserver 전에 celery 구동
```
celery -A config worker --loglevel=info
```
4. 다른 terminal 열어서 장고 runserver 진행
