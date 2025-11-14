# 노트북 1mic 마이크로 진행
# 의도 : 특정 dB이상 넘어가면 telegram로 소음 발생했음을 전송함
#사용법 : 텔레그램 토큰이랑 id 받기 

"""🔑 텔레그램 봇 토큰 받는 법
@BotFather 찾기

텔레그램 앱을 켜고 검색창에 @BotFather 를 검색해서 채팅을 시작해. (옆에 공식 인증 마크가 있는 계정인지 꼭 확인해!)

봇 생성 명령

채팅창에 /newbot 이라고 입력하고 엔터.

BotFather가 "네 봇 이름을 뭐라고 할래?"라고 물어보면, 원하는 이름(예: 소음측정기)을 아무거나 입력해.

사용자 이름 설정

다음에 "네 봇의 사용자 이름(Username)을 뭐라고 할래?"라고 물어볼 거야.

사용자 이름은 반드시 bot으로 끝나야 해! (예: parrot_noise_monitor_bot)

토큰 확인

성공적으로 봇이 만들어지면, BotFather가 축하 메시지와 함께 긴 토큰을 줄 거야.

그 메시지 안에 Use this token to access the HTTP API: 다음에 나오는 긴 문자열이 바로 네 코드에 넣을 TELEGRAM_TOKEN 이야. (예: 1234567890:ABC-DEF123456... 이런 형태)"""

import sounddevice as sd
import numpy as np
import time
import datetime
from scipy.signal import butter, lfilter
import requests # [추가됨] 서버 전송용

# ============================================================
# [앱 알림: Telegram 설정]
# ============================================================
TELEGRAM_TOKEN = "8547094808:AAHe8wyvlpTJ7vwN_aLO42OwM6L5CoxiZZE" # 👈 텔레그램 토큰 (필수)
TELEGRAM_CHAT_ID = "7751976857"          # 👈 텔레그램 채팅 ID (필수)


# ============================================================
# [시뮬레이션] 노트북 마이크 설정
# ============================================================
try:
    default_device = sd.query_devices(kind='input')
    FS = int(default_device['default_samplerate'])
    print(f"✅ 노트북 기본 마이크 감지됨. 샘플링 레이트: {FS} Hz")
except Exception as e:
    print(f"⚠️ 오디오 장치 감지 실패: {e}. 44100Hz로 기본 설정.")
    FS = 44100 # 안전한 기본값

CHUNK = 256

# ============================================================
# [DSP: 대역통과필터(40~250Hz)]
# ============================================================
b, a = butter(4, [40/(FS/2), 250/(FS/2)], btype='band')

dbfs_current = -100.0
last_event_time = 0
event_log = []


# ============================================================
# [네트워크: 층간소음 이벤트 텔레그램 알림 전송]
# 텔레그램 응답 코드를 출력하도록 수정됨
# ============================================================
def send_telegram_alert(timestamp, db_level, threshold):
    try:
        # 메시지 내용 만들기
        message = (f"🚨 층간소음 알림 🚨\n\n"
                   f"시간: {timestamp}\n"
                   f"측정 레벨: {db_level:.1f} dBFS\n"
                   f"기준: {threshold:.1f} dBFS\n\n"
                   f"장치: LAPTOP_SIM_01")
        
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        }
        
        # 텔레그램 API로 HTTP POST 요청 전송 (앱 푸시)
        response = requests.post(telegram_url, data=payload, timeout=1.5)
        print(f"[DEBUG] 텔레그램 응답 코드: {response.status_code}") # 👈 응답 코드 출력
        
    except requests.exceptions.RequestException as e: # 👈 에러 메시지를 잡음
        print(f"[DEBUG] 네트워크 전송 오류 발생: {e}")


# ============================================================
# [시뮬레이션: 시간 기반 자동 조건 처리]
# ============================================================
def get_dbfs_threshold():
    now = datetime.datetime.now().time()
    if now >= datetime.time(22,0) or now < datetime.time(6,0):
        return -30.0, "야간 (시뮬레이션)" # 야간 dBFS 임계값
    else:
        return -25.0, "주간 (시뮬레이션)" # 주간 dBFS 임계값


# ============================================================
# [DSP + Embedded]
# 실시간 오디오 데이터 처리 콜백 함수 (로직 동일)
# ============================================================
def audio_callback(indata, frames, time_info, status):
    global dbfs_current
    data = indata[:, 0]

    # ===== [DSP] 층간소음 대역만 통과해 SNR 증가 =====
    filtered = lfilter(b, a, data)

    # ===== [DSP] RMS → dBFS 변환 =====
    rms = np.sqrt(np.mean(filtered**2) + 1e-12)
    dbfs_current = 20 * np.log10(rms)


# ============================================================
# [시뮬레이션: 노트북 기본 마이크 입력 스트림]
# ============================================================
stream = sd.InputStream(
    samplerate=FS,
    blocksize=CHUNK,
    channels=1,
    callback=audio_callback,
    latency='low',
    dtype='float32'
)

# ============================================================
# 메인 루프
# ============================================================
with stream:
    print("\n=== 노트북 마이크 시뮬레이션 시작 (텔레그램 알림 모드) ===")
    while True:
        dbfs = dbfs_current

        # ===== 기준 자동 선택 (dBFS 기준) =====
        legal_th, day_type = get_dbfs_threshold()

        # ===== bar 표시 =====
        bar_len = int(np.interp(dbfs, [-60, 0], [0, 60]))
        bar = "█" * max(0, min(bar_len, 60))

        print("\033[2J\033[H", end="") # 터미널 청소
        print("=== Real-time Floor Noise Monitor (SIMULATION Mode) ===\n")
        print(f"시간대: {day_type} | 시뮬레이션 기준: {legal_th:.1f} dBFS")
        
        # dBA 표시 제거, dBFS만 표시
        print(f"[{bar:<60}] {dbfs:6.1f} dBFS\n") 

        # ============================================================
        # [판단 로직 (dBFS 기준)]
        # ============================================================
        if dbfs >= legal_th:
            print(f"[DEBUG] 알림 조건 충족! (측정: {dbfs:.1f} dBFS)") # 👈 조건 충족 확인
            print(f"⚠⚠ 층간소음 (시뮬레이션 기준 {legal_th:.1f} dBFS 초과!) ⚠⚠")

            if time.time() - last_event_time > 1: # 1초당 1회만 기록/전송
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                log_msg = f"{timestamp} | {dbfs:.1f} dBFS | 기준 {legal_th:.1f} dBFS 초과"
                event_log.append(log_msg)
                
                last_event_time = time.time()
                
                # ===== [수정] 텔레그램 알림 전송 (앱 푸시) =====
                send_telegram_alert(timestamp, dbfs, legal_th)
                # ============================================

                if len(event_log) > 20:
                    event_log.pop(0)
        else:
            print("(정상 소음 수준)")

        # ===== 이벤트 로그 표시 =====
        print("\n=== 이벤트 로그 (최근 20개) ===")
        if len(event_log) == 0:
            print("아직 이벤트 없음.")
        else:
            for log in event_log:
                print(log)

        time.sleep(0.02)
        
