from imutils import face_utils
from datetime import datetime
import numpy as np
import imutils
import dlib
import cv2
import os
from PIL import Image, ImageFont, ImageDraw
import pygame
import threading
import time
from flask import Flask, render_template, Response, request, jsonify
from tkinter import Tk, Button

# ================================================================
# 초기값 설정
# ================================================================
g_pre_alarm_time = 0
g_window_Size = 30
g_data = []
g_blinkCounter = 0
eye_initial_data = []
threshold_ready = False
g_hide = 0

g_ear_threshold = 0  # 초기 졸음 감지 기준값 (동적으로 설정)

# 한글 폰트 설정 
fontpath = r"C:\Windows\Fonts\gulim.ttc"  # raw string으로 처리하여 경고 제거
font = ImageFont.truetype(fontpath, 20)

# 얼굴 감지 모델 파일 경로 설정
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)  # 파일 경로가 맞는지 확인하세요

# ================================================================
# 눈동자 가로, 세로 비율의 Moving Average 처리
# ================================================================
def calculate_average(value):
    global g_window_Size, g_data

    g_data.append(value)
    if len(g_data) > g_window_Size:
        g_data = g_data[-g_window_Size:]

    if len(g_data) < g_window_Size:
        return 0.0
    return float(sum(g_data) / g_window_Size)

# ================================================================
# 눈동자 가로, 세로 euclidean 거리 구하기
# ================================================================
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# 눈의 가로, 세로 종횡비 구하기 
def eye_aspect_ratio(eye):
    # 눈의 세로 
    a = euclidean_dist(eye[1], eye[5])
    b = euclidean_dist(eye[2], eye[4])
    # 눈의 가로 
    c = euclidean_dist(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)  # 개선된 버전: 평균 계산을 정확히 하기 위해 나누기 2
    return ear

# ================================================================
# 졸음 감지 시 알림 처리 sms send, wave play
# ================================================================
def alarm_notification(filename):
    print("Play wave")
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    # 알람 끄기 버튼 생성
    root = Tk()
    root.title("알람 끄기")
    root.geometry("200x100")

    def stop_alarm():
        pygame.mixer.music.stop()
        root.destroy()

    Button(root, text="알람 끄기", command=stop_alarm).pack(expand=True)
    root.mainloop()

    pygame.quit()

# ================================================================
# Alarm 처리 Thread
# 마지막 알람 발생 후 30초 후 알람 발생 
# ================================================================
def start_Alarm():
    global g_pre_alarm_time
    cur_time = time.time()

    if (cur_time - g_pre_alarm_time) > 30:
        filename = 'test.wav'
        thread = threading.Thread(target=alarm_notification, args=(filename, ))
        thread.start()
        g_pre_alarm_time = cur_time
    else:
        print("Alarm is not progress time: {0}s.".format(int(cur_time - g_pre_alarm_time)))

# ================================================================
# Flask 앱 설정
# ================================================================
app = Flask(__name__)

# 오른쪽, 왼쪽 눈 좌표 인덱스
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ================================================================
# 웹캠 프레임 생성 함수
# ================================================================
def generate_frames():
    global threshold_ready, g_ear_threshold, g_blinkCounter, g_hide, start_time
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 비디오 파일을 열 수 없습니다.")
        exit()

    start_time = time.time()

    while True:
        # 웹캠 영상 읽기
        ret, frame = cap.read()
        if not ret:
            print("비디오가 끝났거나 에러가 발생했습니다.")
            break

        frame = imutils.resize(frame, width=720)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 Detection
        rects = detector(gray)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 왼쪽 및 오른쪽 눈 좌표를 추출하여 양쪽 눈의 눈 종횡비 계산
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            ear_avg = calculate_average(ear)

            # 초기 3초 동안 눈 크기 데이터 수집
            if ear_avg > 0.05:
                if not threshold_ready:
                    img_pillow = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pillow, 'RGBA')
                    draw.text((5, 10), "초기 눈 크기 설정 중입니다. 정면을 바라봐주세요.", 
                              (255, 0, 255), font=font)
                    frame = np.array(img_pillow)
                    elapsed_time = time.time() - start_time
                    eye_initial_data.append(ear)
                    cv2.drawContours(frame, [cv2.convexHull(leftEye)], 0, (255, 0, 255), 1)
                    cv2.drawContours(frame, [cv2.convexHull(rightEye)], 0, (255, 0, 255), 1)
                    if elapsed_time > 10:
                        g_ear_threshold = np.mean(eye_initial_data) - (np.mean(eye_initial_data) / 10)
                        threshold_ready = True
                        print(f"초기 눈 크기 평균값: {np.mean(eye_initial_data)}, 설정된 임계값: {g_ear_threshold}")
                    break

            # 양쪽 눈동자 녹색 외곽선 그리기
            ear_avg = calculate_average(ear)
            if threshold_ready and ear_avg < g_ear_threshold:
                g_blinkCounter += 1
                if g_blinkCounter >= 40:
                    # 졸음 감지 시 양쪽 눈동자 빨간 외곽선 그리기
                    cv2.drawContours(frame, [cv2.convexHull(leftEye)], 0, (0, 0, 255), 1)
                    cv2.drawContours(frame, [cv2.convexHull(rightEye)], 0, (0, 0, 255), 1)
                    img_pillow = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pillow, 'RGBA')
                    draw.text((5, 10), "졸음이 감지 되었습니다", (0, 0, 255), font=font)
                    frame = np.array(img_pillow)
                    start_Alarm()
            else:
                # 양쪽 눈동자 녹색 외곽선 그리기 (정상 상태)
                cv2.drawContours(frame, [cv2.convexHull(leftEye)], 0, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(rightEye)], 0, (0, 255, 0), 1)
                g_blinkCounter = 0
                g_hide = 0

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ================================================================
# Flask 라우팅 설정
# ================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_classes', methods=['POST'])
def update_classes():
    global desired_classes
    desired_classes = request.json.get('desired_classes', [])
    print(f"Updated desired_classes: {desired_classes}")  # 디버깅을 위한 출력
    return jsonify({'status': 'success', 'classes': desired_classes})

# ================================================================
# 메인 실행 부분
# ================================================================
if __name__ == "__main__":
    app.run(debug=True)