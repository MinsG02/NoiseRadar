# 노트북 1mic 마이크로 진행
# 의도 : 특정 dB이상 넘어가면 webhook 사이트로 소음 발생했음을 전송함
#사용한 사이트 : https://webhook.site/#!/view/d8f3e6b4-a8fc-4d1c-8e48-2bec7c66cb0e/7e35abf8-1ca5-4e2f-a6c9-ac4713aa761c/1
#사용한 명령어
#py -m pip install numpy sounddevice scipy requests        
#py webhook_linked_1mic.py
import sounddevice as sd
import numpy as np
import time
import datetime
from scipy.signal import butter, lfilter
import requests # [추가됨] 서버 전송용

# ============================================================
# [시뮬레이션] 서버 URL (앵무가 만든 실제 서버 주소로 변경)
# ============================================================
SERVER_URL = "https://webhook.site/d8f3e6b4-a8fc-4d1c-8e48-2bec7c66cb0e" 

# ============================================================
# [시뮬레이션] 노트북 마이크 설정
#
# ✓ 48000Hz 고정 대신, 노트북의 기본 마이크 샘플링 레이트(FS)를
#   자동으로 가져와서 사용 (예: 44100Hz)
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
# [시뮬레이션] dBA 보정값 제거
#
# INMP441과 달리 노트북 마이크는 감도(Offset)를 알 수 없음.
# 따라서 dBA 변환은 불가능하며, 'dBFS' (디지털 신호) 기준으로만
# 상대적인 크기를 측정.
# ============================================================
# DBFS_TO_DBA_OFFSET = 72  <- (제거됨)


# ============================================================
# [DSP: 대역통과필터(40~250Hz)]
# (동작 원리는 동일)
# FS(샘플링레이트)가 확정된 후에 필터 계수 계산
# ============================================================
b, a = butter(4, [40/(FS/2), 250/(FS/2)], btype='band')

dbfs_current = -100.0
last_event_time = 0
event_log = []


# ============================================================
# [네트워크: 층간소음 이벤트 서버 전송]
# (수정됨: dba_level 대신 db_level(dBFS)을 전송)
# ============================================================
def send_noise_event_to_server(timestamp, db_level, threshold):
    try:
        payload = {
            "timestamp": timestamp,
            "measured_dbfs": db_level,    # dBA -> dBFS로 변경
            "threshold_dbfs": threshold,  # dBA -> dBFS로 변경
            "device_id": "LAPTOP_SIM_01" # 기기 식별자
        }
        response = requests.post(SERVER_URL, json=payload, timeout=1.0)
        
    except requests.exceptions.RequestException as e:
        pass # 오류가 나도 메인 루프는 계속 돌아야 함


# ============================================================
# [시뮬레이션: 시간 기반 자동 조건 처리]
#
# ✓ dBA (절대값) 대신 dBFS (상대값) 기준으로 변경
# ✓ (예: 주간 -25.0 dBFS, 야간 -30.0 dBFS)
# ※ 이 값은 노트북 마이크 민감도에 따라 앵무가 직접 조절해야 함
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
#
# ✓ I²S 대신 시스템의 기본 입력 장치(device=None) 사용
# ✓ 동적으로 감지한 FS 사용
# ============================================================
stream = sd.InputStream(
    samplerate=FS,
    blocksize=CHUNK,
    channels=1,
    callback=audio_callback,
    latency='low',
    dtype='float32'
    # device=None (기본값) -> 시스템 기본 마이크 사용
)

# ============================================================
# 메인 루프
# (수정됨: dBA 변환 로직 제거, dBFS 기준으로 모든 것 처리)
# ============================================================
with stream:
    print("\n=== 노트북 마이크 시뮬레이션 시작 ===")
    while True:
        dbfs = dbfs_current

        # ===== [수정] dBA 근사 변환 (제거) =====
        # dba = dbfs + DBFS_TO_DBA_OFFSET (제거)

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
            print(f"⚠⚠ 층간소음 (시뮬레이션 기준 {legal_th:.1f} dBFS 초과!) ⚠⚠")

            if time.time() - last_event_time > 1: # 1초당 1회만 기록/전송
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                log_msg = f"{timestamp} | {dbfs:.1f} dBFS | 기준 {legal_th:.1f} dBFS 초과"
                event_log.append(log_msg)
                
                last_event_time = time.time()
                
                # ===== [추가] 웹사이트로 이벤트 전송 (dBFS 값 전송) =====
                send_noise_event_to_server(timestamp, dbfs, legal_th)
                # =======================================================

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
