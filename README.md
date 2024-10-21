### **[배리어프리 키오스크]**
프로젝트 기간 : 2024.10.01 ~ 2024.11.15
- **프로젝트 인원 :** 3명
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
    - 손동작 이미지를 전송 받아 학습된 LSTM 모델을 통해 손동작을 판별
    - 판별된 손동작의 결과(손동작, 정확도)를 JSON형태로 Unity로 전송
    - Unity에서 Flask로부터 전송받은 JSON 객체를 통해 동작
- **보완할점**
    - Seq2Seq 모델 학습을 위한 데이터셋 부족(소상공인 질의-응답 데이터를 사용)
    - Seq2Seq모델 학습시 mecab과 같은 형태소분석을 추가, attention이나 transformor 모델을 사용하여 성능 향상 가능
 
<hr>

- **프로젝트 과정**
  - **데이터 전처리**
    - **사용한 데이터셋:** [AI허브-소상공인 고객 주문 질의-응답 텍스트](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=102)
    - ![비정상 데이터](https://github.com/user-attachments/assets/ced4955b-284e-4d77-9dea-bc3f4e83bd11) 원본 데이터는 사진과 같이 비정상 데이터 존재 및 키오스크와 관련 없는 데이터(예약 등)이 존재
       
