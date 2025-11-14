#실시간 소음 파라미터를 보고 delay 파악하기 위함 (사용X)
import sounddevice as sd
import numpy as np
import datetime
from scipy.signal import butter, lfilter

# ===== 층간소음 기준 =====
def get_noise_threshold():
    now = datetime.datetime.now().time()
    if now >= datetime.time(22,0) or now < datetime.time(6,0):
        # 야간
        return 34  
    else:
        # 주간
        return 39

# ===== 필터 =====
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def apply_filter(data, lowcut=40, highcut=250, fs=48000):
    b, a = butter_bandpass(lowcut, highcut, fs, order=4)
    return lfilter(b, a, data)

# ===== dB 측정 =====
def calc_db(data):
    rms = np.sqrt(np.mean(data**2)) + 1e-12
    db = 20 * np.log10(rms)
    return db

# ===== 방향 판별 (임시: 에너지 기반) =====
def estimate_direction(data):
    fft = np.abs(np.fft.rfft(data))
    low = np.sum(fft[50:200])      # 저주파대
    mid = np.sum(fft[200:500])     # 중간
    high = np.sum(fft[500:1200])   # 고주파대

    if high > low * 1.5:
        return "Right"
    elif low > high * 1.5:
        return "Left"
    else:
        return "Center"

# ===== 실시간 이벤트 감지 =====
FS = 48000
CHUNK = 2048

stream = sd.InputStream(
    channels=1,
    samplerate=FS,
    device=None,     # INMP441 장치명 필요하면 수정
    blocksize=CHUNK
)

print("▶ 실시간 층간소음 감지 시작")
print("TigerVNC 터미널에서 실시간 모니터링 가능\n")

with stream:
    while True:
        frame, _ = stream.read(CHUNK)
        frame = frame[:, 0]  # 1채널

        # 필터 적용
        filtered = apply_filter(frame, 40, 250, FS)

        # dB 측정
        db = calc_db(filtered)

        # 층간소음 기준 가져오기 (주간/야간 자동)
        TH = get_noise_threshold()

        # 실시간 막대 표시
        bar = "█" * int(max(0, db + 60) / 2)
        print(f"\r[{bar:<60}] {db:5.1f} dB  (기준 {TH} dB)", end="")

        # ▣ 이벤트 발생
        if db >= TH:
            direction = estimate_direction(filtered)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n⚠️ 이벤트 발생! {db:.1f} dB  방향: {direction}  시간: {now}\n")
