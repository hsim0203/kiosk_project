import whisper
import pyaudio
import wave
from flask import Flask,jsonify
from flask import request
from werkzeug.utils import secure_filename
import tensorflow as tf
import json
import base64

#seq2seq관련
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#손동작 인식 관련
from tensorflow import keras
import mediapipe as mp
import numpy.linalg as LA
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#메모리 캐시 비우기
import torch
#시간
from datetime import datetime

#메모리 할당방식 변경
#tensorflow와 keras는 메모리 파편화 방지를 위해 GPU 메모리를 최대한 매핑해 메모리 초과문제 발생
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,',len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

#메모리 캐시 비우기
torch.cuda.empty_cache()

########################seq2seq모델 관련########################
#seq2seq모델 로드(모델,인코더모델, 디코더모델)
seq2seq_model = tf.keras.models.load_model('C:/deep_learning_project01/seq2seq_model/seq2seq_1010_3_model.h5')
encoder_seq2seq_model = tf.keras.models.load_model('C:/deep_learning_project01/seq2seq_model/seq2seq_1010_3_encoder_model.h5')
decoder_seq2seq_model = tf.keras.models.load_model('C:/deep_learning_project01/seq2seq_model/seq2seq_1010_3_decoder_model.h5')
#tokenizer 열기
tokenizer2 = None
# Tokenizer 저장
with open('C:/deep_learning_project01/seq2seq_model/tokenizer_1010_3.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)
#print(tokenizer2.word_index)
#max_len 변수들을 따로 pickle로 저장하지 않았으므로 해당 변수 선언
max_len_a2 =36
max_len_q2 =21

#디코딩 함수 정의
def decode_sequence(input_seq):
    # 인코더 상태 추출
    states_value = encoder_seq2seq_model.predict(input_seq)

    # start 토큰으로 시작하는 타겟 시퀀스
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer2.word_index['start']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_seq2seq_model.predict([target_seq] + states_value)

        # 예측된 토큰을 텍스트로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer2.index_word.get(sampled_token_index, '')
        decoded_sentence += ' ' + sampled_word

        # end 토큰을 만나거나 일정 길이를 넘으면 중단
        if (sampled_word == 'end' or len(decoded_sentence) > max_len_a2):
            stop_condition = True

        # 타겟 시퀀스 업데이트 (다음 단계 예측을 위해)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # 상태 업데이트
        states_value = [h, c]

    return decoded_sentence
###########################################################

########################whisper 모델########################
whisper_model = whisper.load_model('medium')
###########################################################

########################손동작 인식 관련######################
#
# # LSTM 모델 경로
# hands_model = keras.models.load_model('C://ai_project01/hand_train_result0930')
# # LSTM모델 출력
# hands_model.summary()
#
# # 손동작 5번째마다 예측
# seq_length = 30
#
# gesture = {
#     0: 'LEFT', 1: "RIGHT", 2: "UP", 3: 'DOWN', 4: 'SELECT'
# }
#
# mp_hands = mp.solutions.hands

##KNN 예측
#csv파일의 내용 출력시 모든 컬럼 출력
pd.set_option('display.max_columns', None)
#csv 파일 내용 출력시 한줄에 모든 컬럼 출력
pd.set_option('display.expand_frame_repr', False)

gesture = {
    0: 'FIRST', 1: "ONE", 2: "TWO", 3: 'THREE', 4: 'FOUR', 5: 'FIVE', 6: 'SIX', 7: 'ROCK', 8: 'SPIDERMAN', 9: 'YEACH',
    10: 'OK',
}
kiosk_gesture = {
    0:'NO', 1:'LEFT', 9:'RIGHT', 6:'UP', 4:'DOWN', 10:'OK'
}
gesture_df = pd.read_csv('c:/ai_project01/whisper-rest-server/gesture_train.csv', header=None)

angle = gesture_df.iloc[:, :-1]
angle_arr = angle.values.astype(np.float32)
label = gesture_df.iloc[:, -1]
label_arr = label.values.astype(np.float32)

knn = cv2.ml.KNearest_create()

knn.train(angle_arr, cv2.ml.ROW_SAMPLE, label_arr)

mp_hands = mp.solutions.hands

###########################################################

#Flask
app = Flask(__name__)
#파일의 최대 저장 용량 설정, 500MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

#음성인식 이후 답변 출력 후 리턴
@app.route('/wavtest', methods=['POST'])
def whisper():

    #POST전  송받은 file
    #f = request.files['file']
    #/files/ 에 파일이름으로 저장
    #f.save('./files/'+secure_filename(f.filename))

    #POST전  송받은 file
    byte_wav = request.data
    print(type(byte_wav))

    # byte[] 데이터를 파일로 저장
    wav_file_path = 'c://ai_project01/whisper-rest-server/files/file.wav'
    with open(wav_file_path, 'wb') as file:
        file.write(byte_wav)

    now = datetime.now()
    nowStr = now.strftime('%Y-%m-%d %H:%M:%S.%f')

    filepath1 = 'c://ai_project01/whisper-rest-server/files/file.wav'
    result = whisper_model.transcribe(filepath1)

    input_question = result['text']  # 새로운 질문 입력
    input_seq = tokenizer2.texts_to_sequences([input_question])
    input_seq = pad_sequences(input_seq, maxlen=max_len_q2, padding='post')

    # 예측
    predicted_sentence = decode_sequence(input_seq)

    print(f"Whisper 음성 인식(질문): {input_question}")
    #print(f"Seq2Seq 답변 생성(응답): {predicted_sentence}")
    
    #end 앞에서 끊기
    result = predicted_sentence.split(' end')[0]
    print(f"Seq2Seq 답변 생성(응답): {result}")
    #print(result)
    print(jsonify(result=result, input_question=input_question, time=nowStr))
    return jsonify(result=result, input_question=input_question, time=nowStr)

##KNN
#손동작 탐지 후 결과 리턴
@app.route("/lstm_detect", methods=["POST"])
def lstm_detect01():
    # 탐지 결과를 저장하는 변수
    gesture_result = []
    all_results = []
    highest_conf_result = None  # 가장 높은 신뢰도를 저장할 변수
    #손, 손가락 위치 탐지 시작
    with mp_hands.Hands() as hands:
        json_image = request.get_json()
        now = datetime.now()
        nowStr = now.strftime('%Y-%m-%d %H:%M:%S.%f')
        encoded_data_arr = json_image.get("data")

        if len(encoded_data_arr) >= 5:
            encoded_data_arr = encoded_data_arr[:5]  # 5개로 제한

        for index, encoded_data in enumerate(encoded_data_arr):
            encoded_data = encoded_data.replace("image/jpeg;base64,", "")
            decoded_data = base64.b64decode(encoded_data)

            nparr = np.frombuffer(decoded_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                #탐지한 keypoint 순서대로 1개씩 저장
                for hand_landmarks in results.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    #j => keypoint의 index, lm : keypoint의 좌표
                    for j, lm in enumerate(hand_landmarks.landmark):
                        #각도를 구하기 위해 x,y,z 좌표 대입
                        joint[j] = [lm.x, lm.y, lm.z]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                    v = v2 - v1
                    #v를 정규화(1차원배열)
                    v_normal = LA.norm(v, axis=1)
                    #v와 연산하기 위해 2차원 배열로 변환
                    v_normal2 = v_normal[:, np.newaxis]
                    #v를 나눠서 거리 정규화
                    v2 = v / v_normal2
                    #a,b 배열의 곱
                    a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
                    b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                    ein = np.einsum('ij,ij->i', a, b)
                    #코사인값 계산
                    radian = np.arccos(ein)
                    #코사인값 각도로 변환
                    angle = np.degrees(radian)

                    #각도 배열로 변환
                    data = np.array([angle], dtype=np.float32)
                    #retval=>손모양 결과 실수로 리턴, results =>손모양 결과 배열
                    #neighbours=>거리가 가장 가까운 손모양 3개, #dist => 가장 가까운 거리 3개
                    retval, results, neighbours, dist = knn.findNearest(data, 3)
                    #예측값 정수로 변환
                    idx = int(retval)

                    # 거리 값을 정규화
                    norm_dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
                    avg_norm_dist = np.mean(norm_dist)

                    conf = (1 - avg_norm_dist) * 1.5 # 확률을 1.5배로 스케일링
                    conf = min(conf, 1.0)# 최대값 1로 제한

                    gesture_text = kiosk_gesture.get(idx, "UNKNOWN")
                    print(gesture_text)

                    if highest_conf_result is None or conf > highest_conf_result['conf']:
                        #gesture_result.append({
                        #    "text": f"{gesture_text}",
                        #    "conf": f"{round(conf * 100, 2)}%",  # 확률을 퍼센트로 변환
                        #    "time": f"{nowStr}"
                        #})

                        #all_results.append({
                        #    "gesture": f"{gesture_text}",
                        #    "conf": f"{round(conf * 100, 2)}%"
                        #})

                        highest_conf_result = {
                            "text": f"{gesture_text}",
                            "conf": conf  # 비교를 위해 raw conf 값을 저장
                            #"x": int(hand_landmarks.landmark[0].x * image.shape[1])
                            #"y": int(hand_landmarks.landmark[0].y * image.shape[0])
                            #,"time": f"{nowStr}"
                        }
    if highest_conf_result:
        highest_conf_result["conf"] = f"{round(highest_conf_result['conf'] * 100, 2)}%"
        #highest_conf_result["conf"] = f"95%"
        gesture_result.append(highest_conf_result)
    print(gesture_result)
    return json.dumps(gesture_result)

##LSTM
#손동작 탐지 후 결과 리턴
# @app.route("/lstm_detect", methods=["POST"])
# def lstm_detect01():
#     # 탐지 결과를 저장하는 변수
#     lstm_result = []
#     # 손동작을 저장하는 리스트
#     seq = []
#
#     # 화면에서 손과 손가락 위치 탐지
#     with mp_hands.Hands() as hands:
#         json_image = request.get_json()
#         now = datetime.now()
#         nowStr = now.strftime('%Y-%m-%d %H:%M:%S.%f')
#
#
#         encoded_data_arr = json_image.get("data")
#
#
#         for index, encoded_data in enumerate(encoded_data_arr):
#
#
#             encoded_data = encoded_data.replace("image/jpeg;base64,", "")
#             decoded_data = base64.b64decode(encoded_data)
#
#             # decoded_data -> 1차원 배열 변환
#             nparr = np.fromstring(decoded_data, np.uint8)
#
#             # nparr -> BGR 3차원 배열 변환
#             image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#
#             # BGR -> RGB로 변환 후 손, 손가락 관절위치 탐지 후 리턴
#             results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#             # results.multi_hand_landmarks - 탐지된 손의 keypoint 값들이 저장
#             if results.multi_hand_landmarks != None:
#
#                 # hand_landmarks에 탐지된 keypoint값을 순서대로 1개씩 저장,
#                 for hand_landmarks in results.multi_hand_landmarks:
#
#                     joint = np.zeros((21, 4))
#
#                     # hand_landmarks.landmark - 손의 keypoint 좌표 리턴
#                     # j - keypoint의 index
#                     # lm - keypoint의 좌표
#                     for j, lm in enumerate(hand_landmarks.landmark):
#                         #print("j=", j)
#                         #print("lm=", lm)
#                         # keypoint의 x,y,z좌표
#                         #print("lm.x", lm.x)
#                         #print("lm.y", lm.y)
#                         #print("lm.z", lm.z)
#                         #print("lm.visibility", lm.visibility)
#                         # 각도를 구하기 위해 x,y,z 좌표 배열에 대입
#                         joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
#                         #print("=" * 100)
#
#                     print("=" * 100)
#                     print("joint=", joint)
#                     print("=" * 100)
#
#                     v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
#                     v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
#
#                     # v1에서 v2를 빼서 거리를 계산
#                     # v는 2차원 배열
#                     v = v2 - v1
#                     print("=" * 100)
#                     print("v=", v)
#                     print("=" * 100)
#
#                     # v를 1차원배열로 정규화
#                     v_normal = LA.norm(v, axis=1)
#                     print("=" * 100)
#                     print("v_normal=", v_normal)
#                     print("=" * 100)
#
#                     # v와 연산을 위해 2차원배열로 변환
#                     v_normal2 = v_normal[:, np.newaxis]
#                     print("=" * 100)
#                     print("v_normal2=", v_normal2)
#                     print("=" * 100)
#
#                     # v/v_normal2 로 나눠서 거리를 정규화
#                     v2 = v / v_normal2
#                     print("=" * 100)
#                     print("v2=", v2)
#                     print("=" * 100)
#
#                     a = v2[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
#                     b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
#
#                     # ein - 행렬곱
#                     # a와 b 배열의 곱 계산
#                     ein = np.einsum('ij,ij->i', a, b)
#                     print("=" * 100)
#                     print("ein=", ein)
#                     print("=" * 100)
#
#                     # radian - 코사인값(1차원 배열)
#                     radian = np.arccos(ein)
#                     print("=" * 100)
#                     print("radian=", radian)
#                     print("=" * 100)
#
#                     # radian값을 각도로 변환
#                     angle = np.degrees(radian)
#                     print("=" * 100)
#                     print("angle=", angle)
#                     print("=" * 100)
#
#                     # joint.flatten() - 관절 좌표를 1차원 배열로 변환
#                     # data에 관절 좌표, 각도 저장
#                     data = np.concatenate([joint.flatten(), angle])
#                     print("=" * 100)
#                     print("data=", data)
#                     print("=" * 100)
#
#                     seq.append(data)
#                     #디버깅
#                     print(f"seq의 길이: {len(seq)}")
#
#                     if len(seq) < seq_length:
#                         continue
#
#                     # last_seq <- 마지막 손동작 행 길이만큼 대입
#                     last_seq = seq[-seq_length:]
#                     # last_seq -> 배열로 변환
#                     input_arr = np.array(last_seq, dtype=np.float32)
#                     print("input_arr=", input_arr.shape)
#
#                     # input_arr을 3차원 배열로 변환
#                     input_lstm_arr = input_arr.reshape(1,30,99)
#                     # 디버깅
#                     print("input_lstm_arr의 shape:", input_lstm_arr.shape)
#                     print("=" * 100)
#                     print("input_lstm_arr=", input_lstm_arr)
#                     print("=" * 100)
#                     print("=" * 100)
#                     print("input_lstm_arr.shape=", input_lstm_arr.shape)
#                     print("=" * 100)
#
#                     # y_pred <- lstm 모델을 통해 수어 예측 후 대입
#                     y_pred = hands_model.predict(input_lstm_arr)
#                     print("=" * 100)
#                     print("y_pred=", y_pred)
#                     print("=" * 100)
#
#                     # idx <- y_pred에서 가장 예측 확률 값이 가장 높은 인덱스 대입
#                     idx = int(np.argmax(y_pred))
#                     print("=" * 100)
#                     print("idx=", idx)
#                     print("=" * 100)
#
#                     letter = gesture[idx]
#                     print("=" * 100)
#                     print("letter=", letter)
#                     print("=" * 100)
#
#                     # conf <- idx번째의 확률 대입
#                     conf = y_pred[0, idx]
#                     print("=" * 100)
#                     print("conf=", conf)
#                     print("=" * 100)
#
#                     # 탐지 결과 lstm_result에 추가
#                     lstm_result.append({
#                         "text": f"{letter}",
#                         "conf":f"{round(conf *100,2)}",
#                         "time":f"{nowStr}"
#                     })
#
#                     print("=" * 100)
#                     print("lstm_result=", lstm_result)
#                     print("=" * 100)
#
#     # 결과를 json으로 변환 후 return
#     return json.dumps(lstm_result)



#####################################테스트####################################

#whisper 음성인식 테스트
@app.route('/wavtest1', methods=['POST'])
def whispertest():

    #POST전  송받은 file
    f = request.files['file']
    #/files/ 에 파일이름으로 저장
    f.save('./files/'+secure_filename(f.filename))

    filepath1 = 'c://ai_project01/whisper-rest-server/files/'+secure_filename(f.filename)
    result = whisper_model.transcribe(filepath1)
    print(result['text'])
    return result["text"]

#seq2seq 테스트
@app.route('/wavtest2', methods=['POST'])
def seq2seq():
    input_question = "아이스 아메리카노 주세요"  # 새로운 질문 입력
    input_seq = tokenizer2.texts_to_sequences([input_question])
    input_seq = pad_sequences(input_seq, maxlen=max_len_q2, padding='post')
    # 예측
    predicted_sentence = decode_sequence(input_seq)

    print(f"질문: {input_question}")
    print(f"예측된 답변: {predicted_sentence}")

    result = predicted_sentence.split(' end')[0]
    return jsonify(result=result)

#파일전송 테스트
@app.route('/wavtest3', methods=['POST'])
def wav_send_test01():
    result = "test"

    byte_wav = request.data
    print(type(byte_wav))

    # byte[] 데이터를 파일로 저장
    wav_file_path = 'c://ai_project01/whisper-rest-server/files/file.wav'
    with open(wav_file_path, 'wb') as file:
        file.write(byte_wav)

    if byte_wav[:4] == b'RIFF' and byte_wav[8:12] == b'WAVE':
        print("데이터는 WAV 파일 형식입니다.")
    else:
        print("올바르지 않은 WAV 파일 형식입니다.")

    try:
        with wave.open(wav_file_path, 'rb') as f:
            print(f.getnchannels(), f.getframerate(), f.getnframes())
    except wave.Error as e:
        print("파일 형식이 올바르지 않습니다:", e)

    print("파일이 성공적으로 저장되었습니다.")
    return result

if __name__ == '__main__':
#    app.run()
    app.run(host='0.0.0.0', port=5000)