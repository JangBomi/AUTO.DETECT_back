# AUTO.DETECT
### <b>도서관 출입 반입 금지 물품 자동 탐지를 위한 - AUTO.DETECT</b>
이화여자대학교 캡스톤디자인프로젝트B 산학트랙 그로쓰17팀 딥딥

## 문제점 분석 및 개발 목표
<img width="754" alt="스크린샷 2021-09-21 오전 1 57 08" src="https://user-images.githubusercontent.com/55357130/134042837-7415a063-001b-4691-9797-e73472bc09fc.png">
자동으로 반입 금지 물품을 검출하고, 기록하고, 경고하는 시스템을 제작


## 주요 기능 및 UI
### 로그인 & 홈
<img width="450" alt="스크린샷 2021-09-21 오전 1 59 27" src="https://user-images.githubusercontent.com/55357130/134043175-52d696cc-124b-4884-b00f-a1b4d1c283b5.png"><img width="450" alt="스크린샷 2021-09-21 오전 1 59 54" src="https://user-images.githubusercontent.com/55357130/134043251-698c260d-1fb3-43f3-9e0e-f41a7c1f66a7.png">

### 실시간 검출 & 저장
<img width="450" alt="스크린샷 2021-09-21 오전 2 02 40" src="https://user-images.githubusercontent.com/55357130/134047347-98489878-c39a-4d27-848e-2187e24f4c60.jpeg"><img width="450" alt="스크린샷 2021-09-21 오전 2 01 42" src="https://user-images.githubusercontent.com/55357130/134045226-c1dc5dc7-e186-45d4-8362-79bd3840f70d.png">
- 컵을 탐지할 때 경고음을 내며 자동으로 저장됨

### 기록 목록
<img width="450" alt="스크린샷 2021-09-21 오전 2 01 54" src="https://user-images.githubusercontent.com/55357130/134045257-5a7f1253-8577-4c09-adcf-b9ab18082e1d.png"><img width="450" alt="스크린샷 2021-09-21 오전 2 02 14" src="https://user-images.githubusercontent.com/55357130/134047975-4fb7b886-aa97-48ea-96b7-2e7e01a12e2c.png">


## 모델 성능 분석
<img width="925" alt="스크린샷 2021-09-21 오전 3 04 57" src="https://user-images.githubusercontent.com/55357130/134051892-ebbb4b5a-7be2-4a9a-8535-801f1a1c1947.png">
<b> 1. 사용자의 다양한 음료 소지 방법에 따른 탐지 성능 강화 </b><br>
- 사용자마다 각기 다른 음료 소지 방법을 가지고 도서관에 입장하기때문에 다양한 각도에서의 탐지 정확도를 향상시키기 위해 이미지를 (ROTATION의) data augmention 기법을 사용하여 이미지 변환<br>
<b> 2. 밝기에 따른 성능 저하 문제 감소를 위한 색 변환</b><br>
- 어두운 상황에서도 판별할 수 있게 학습시키기 위해 기존 이미지에 grayscale색 random 변환 작업을 사용<br>
<b>3. 작은 객체와 빠른 객체 탐지를 위한 모델 커스텀</b><br>
- 작은 객체 탐지를 위해 이미지를 (ZOOM, CUT의) data augmentation기법을 사용하여 이미지 변환 & 코드변형/ 빠른 객체 탐지를 위해 모델 연산 부분 코드 수정<br>


### 모델 커스텀 전 후 비교
![ezgif com-video-to-gif-3](https://user-images.githubusercontent.com/55357130/134056745-37500df3-941f-4db0-a735-21b925a17f60.gif) ![ezgif com-video-to-gif-2](https://user-images.githubusercontent.com/55357130/134055425-ae4f43f0-1ed9-4d0b-b074-3379b27cff4a.gif)
<br>
모델 커스텀 전 　　　　　　　　　　　　　　　　　모델 커스텀 후


## 기대효과
<img width="507" alt="스크린샷 2021-09-21 오전 2 36 01" src="https://user-images.githubusercontent.com/55357130/134048045-f40b948a-0529-4c6f-aad3-1d871778f7a1.png">
<b> 다양한  </b>


## 사용 기술 및 구조
<img height="110" alt="스크린샷 2021-09-21 오전 2 49 03" src="https://user-images.githubusercontent.com/55357130/134049819-42b50a3c-94e9-4e71-872d-71df6a5fb3eb.png"><img height="110" alt="스크린샷 2021-09-21 오전 2 49 23" src="https://user-images.githubusercontent.com/55357130/134049855-d3bf20e8-e040-4440-98d7-558f1da5e9cb.png"><img height="110" alt="스크린샷 2021-09-21 오전 2 50 01" src="https://user-images.githubusercontent.com/55357130/134049942-8713719e-3b6c-48cc-a2f4-98104fb9131b.png"><img height="110" alt="스크린샷 2021-09-21 오전 2 50 32" src="https://user-images.githubusercontent.com/55357130/134050028-18479146-8e7f-48eb-b7d3-14289f9ddb9c.png"><img height="110" alt="스크린샷 2021-09-21 오전 2 51 06" src="https://user-images.githubusercontent.com/55357130/134050084-f2d60696-266b-4db8-9f7f-70deea17397c.png">


<img width="719" alt="스크린샷 2021-09-21 오전 2 38 38" src="https://user-images.githubusercontent.com/55357130/134048432-8c46c52e-a09a-401a-bc17-aaa086258ed7.png">


## 개발기간
2020.09 ~ 2021.05

## 팀원
남궁지윤 -  관련 기술 조사 및 데이터 정리<br>
양은서 - Yolov4모델을 이용한 실시간 검출 모델 개발<br>
이경민 - Yolov4모델을 이용한 실시간 검출 모델 개발<br>
장보미 - 웹 프론트/백 개발<br>


## 자료
### 발표 ppt
https://docs.google.com/presentation/d/1LcyZIhmRqekXAoI6hOrGUtof5E4FhjgC/edit?usp=sharing&ouid=116372835815595992183&rtpof=true&sd=true
### 포스터
[졸업프로젝트-포스터-그로쓰17 FINAL.pdf](https://github.com/JangBomi/AUTO.DETECT_front/files/7197738/-.-.17.FINAL.pdf)
### 시연 영상
https://docs.google.com/presentation/d/1LcyZIhmRqekXAoI6hOrGUtof5E4FhjgC/edit?usp=sharing&ouid=116372835815595992183&rtpof=true&sd=true







# 패키지 다운
pip install -r requirements.txt
<br>

## requirements.txt 생성
pip freeze > requirements.txt

# 시작
python manage.py runserver

# 마이그레이션 파일 생성
python manage.py makemigrations <app-name>

# 마이그레이션 적용
python manage.py migrate <app-name>

# 마이그레이션 적용 현황 
python manage.py showmigrations <app-name>

# 지정 마이그레이션의 SQL 내역
python manage.py sqlmigrate <app-name> <migration-name>

# db 실행
./manage.py dbshell
