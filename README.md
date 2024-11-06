### **[배리어프리 키오스크]**
프로젝트 기간 : 2024.10.01 ~ 2024.11.08
- **프로젝트 인원 :** 2명
- **담당 업무:** 백앤드 개발(데이터셋 생성 및 전처리, ai 모델 학습, Rest-server구현)
- **개발 환경:** Anaconda Python 3.10, tensorflow-gpu 2.10, Mediapipe0.10.14, protobuf 3.19.4, Numpy 1.23.5, torch 2.3.0, Flask
- **개요 및 목적**
    - 키오스크 사용이 힘들거나 불편한 사람들을 위한 배리어 프리 키오스크 제작
    - 딥러닝 모델에 대한 학습 및 이해도 향상
    - 백앤드, 프론트앤드를 확실히 구분해 개발하는 방법 학습
- **세부기능**
  - **음성인식**
    - Unity로부터 음성파일을 전송받은 후 Whisper를 사용해 text로 변경
    - Whisper를 통해 변환된 text를 seq2seq모델에 입력 시퀀스로 사용해 출력 시퀀스 출력
    - 출력 시퀀스를 답변으로 하여 Unity로 넘겨주고 tts처리를 통해 대화형 키오스크 구현
  - **손동작 인식**
    - 손동작 이미지를 전송 받아 학습한 KNN알고리즘을 통해 손동작을 판별
    - 판별된 손동작의 결과(손동작, 정확도)를 JSON형태로 Unity로 전송
    - Unity에서 Flask로부터 전송받은 JSON 객체를 통해 동작
    - ~~손동작 이미지를 전송 받아 학습된 LSTM 모델을 통해 손동작을 판별~~(판별을 하기위해 프레임이 많이 필요하고, 응답시간이 길어 KNN으로 대체)
- **보완할점**
    - Seq2Seq 모델 학습을 위한 데이터셋 부족(소상공인 질의-응답 데이터를 사용)
    - Seq2Seq모델 학습시 mecab과 같은 형태소분석을 추가, attention이나 transformor 모델을 사용하여 성능 향상 가능
 
<hr>

- **동작 예시**
  - 음성인식 후 답변 생성
  - ![음성답변](https://github.com/user-attachments/assets/fdd25b13-de30-4a40-9301-d732ed3e4411)
  - 손동작 인식 후 동작판별
  - ![손동작 제스쳐]("https://github.com/user-attachments/assets/5b06fd66-0a7d-4d86-a777-a158e7c01bb1")
  <!-- - ![손동작](https://github.com/user-attachments/assets/fd820497-1833-4aed-bf28-12b7245fc748) -->


 

<hr>

- **프로젝트 과정**
  - **음성 데이터 전처리**
    - **사용한 데이터셋:** [AI허브-소상공인 고객 주문 질의-응답 텍스트](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=102) 169,633행 데이터
    - 키오스크의 특성상 고객 질의에 응답하는 방식으로 시작
     ![비정상 데이터](https://github.com/user-attachments/assets/ced4955b-284e-4d77-9dea-bc3f4e83bd11)
    - 원본 데이터는 사진과 같이 Q/A가 연달아 나오지 않는 비정상 데이터 존재 및 키오스크와 관련 없는 데이터(예약 등)이 존재
    - 비정상 데이터 및 불필요 데이터 삭제, 고객이 발화자이고 Q인 대화문, 발화자가 가게이고 A인 대화문으로 분리
    ![전처리1](https://github.com/user-attachments/assets/c34e18ad-b541-4d2b-bb86-a6fc7e561dcd)
    - 새로운 인덱스 번호를 붙인 후 Q와 A 데이터 한 행으로 병합
    ![전처리2](https://github.com/user-attachments/assets/bbf2dd7f-b207-4b3c-a145-c71086c56bd8)
    - **토크나이징:** 데이터 Q/A로 분리 후 STRAT, END토큰을 답변에 추가 후 tokenizer 설정
    - **패딩:** 각 Q,A의 최대 길이에 맞춰 post_padding
  - **Sequence to Sequence 모델 학습**
  -  ~~**손동작 데이터셋 생성**~~
  -  ~~**손동작 데이터셋 전처리**~~
  -  ~~**손동작 LSTM 모델 학습**~~
  -  **Mediapipe를 통한 KNN 손동작 알고리즘**
    - [ntu-rris/google-mediapipe](https://github.com/ntu-rris/google-mediapipe) gesture 데이터 사용
    - 손 관절의 정보를 추출해 각도를 계산해 손동작을 분류
  -  **Rest-server 구현(Flask)**
