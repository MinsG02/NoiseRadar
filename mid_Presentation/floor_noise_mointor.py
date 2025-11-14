import sounddevice as sd
import numpy as np
import time
import datetime   # ← 핵심 수정 (모듈 전체 import)
from scipy.signal import butter, lfilter

# === 보정값 (너 환경 기준) ===
DBFS_TO_DBA_OFFSET = 72 # -50 + 72 → 22 dBA

FS = 48000
CHUNK = 256

# 필터 준비
b, a = butter(4, [40/(FS/2), 250/(FS/2)], btype='band')

dbfs_current = -100.0
last_event_time = 0
event_log = []


# ===== 주간/야간 층간소음 기준 가져오기 =====
def get_legal_threshold():
    now = datetime.datetime.now().time()   # ← 수정됨
    if now >= datetime.time(22,0) or now < datetime.time(6,0):
        return 34, "야간"
    else:
        return 39, "주간"


def audio_callback(indata, frames, time_info, status):
    global dbfs_current
    data = indata[:, 0]

    filtered = lfilter(b, a, data)
    rms = np.sqrt(np.mean(filtered**2) + 1e-12)
    dbfs_current = 20 * np.log10(rms)


stream = sd.InputStream(
    samplerate=FS,
    blocksize=CHUNK,
    channels=1,
    callback=audio_callback,
    latency='low',
    dtype='float32'
)

with stream:
    while True:
        dbfs = dbfs_current

        # === dBA로 변환 ===
        dba = dbfs + DBFS_TO_DBA_OFFSET

        # === 법적 기준 가져오기 ===
        legal_th, day_type = get_legal_threshold()

        # bar(dBFS 기반)
        bar_len = int(np.interp(dbfs, [-60, 0], [0, 60]))
        bar = "█" * max(0, min(bar_len, 60))

        print("\033[2J\033[H", end="")

        print("=== Real-time Floor Noise Monitor (Ultra Low Latency) ===\n")
        print(f"시간대: {day_type}  |  법적 기준: {legal_th} dBA")
        print(f"[{bar:<60}]  {dbfs:6.1f} dBFS  /  {dba:6.1f} dBA\n")

        # ===== 법적 기준 초과 여부 판정 =====
        if dba >= legal_th:
            print("⚠⚠  층간소음 기준 초과! (법적 기준 위반 가능) ⚠⚠")

            if time.time() - last_event_time > 1:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                event_log.append(
                    f"{timestamp} | {dba:.1f} dBA | 기준 {legal_th} dBA 초과"
                )
                last_event_time = time.time()

                if len(event_log) > 20:
                    event_log.pop(0)
        else:
            print("(정상 소음 수준)")

        # ===== 이벤트 로그 출력 =====
        print("\n=== 이벤트 로그 (최근 20개) ===")
        if len(event_log) == 0:
            print("아직 이벤트 없음.")
        else:
            for log in event_log:
                print(log)

        time.sleep(0.02)
