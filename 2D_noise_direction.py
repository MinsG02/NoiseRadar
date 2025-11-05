import time
import math
import numpy as np
import sounddevice as sd
import board, busio, csv
from scipy.signal import butter, lfilter
from digitalio import DigitalInOut, Direction
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
import matplotlib.pyplot as plt
from collections import deque

# ==== 설정 ====
RATE = 48000           # 샘플링 레이트
CHUNK = 2048           # 버퍼 크기
THRESHOLD_DB = 60      # 감지 임계 데시벨
SNR_MIN = 10           # 최소 SNR
LOWCUT, HIGHCUT = 40, 250  # 층간소음 대역
CSV_LOG = "floor_noise_log.csv"

# ==== OLED 초기화 ====
i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)
font = ImageFont.load_default()

# ==== LED 설정 ====
led_left = DigitalInOut(board.D17)
led_right = DigitalInOut(board.D27)
led_left.direction = Direction.OUTPUT
led_right.direction = Direction.OUTPUT

# ==== 필터 함수 ====
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

# ==== 기본 유틸 ====
def rms_db(signal):
    rms = np.sqrt(np.mean(np.square(signal)))
    if rms <= 1e-8:
        return -100
    return 20 * np.log10(rms)

def snr_db(signal):
    noise_floor = np.mean(np.abs(signal)) * 0.2
    power_signal = np.mean(signal ** 2)
    power_noise = noise_floor ** 2
    if power_noise == 0:
        return 100
    return 10 * np.log10(power_signal / power_noise)

def gcc_phat(sig1, sig2):
    n = sig1.shape[0] + sig2.shape[0]
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R /= np.abs(R) + 1e-15
    cc = np.fft.irfft(R, n=n)
    max_shift = int(n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / float(RATE)

def show_text(text):
    image = Image.new("1", (oled.width, oled.height))
    draw = ImageDraw.Draw(image)
    draw.text((5, 25), text, font=font, fill=255)
    oled.image(image)
    oled.show()

# ==== 로그 파일 초기화 ====
with open(CSV_LOG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "dB(L)", "dB(R)", "SNR(L)", "SNR(R)", "Direction"])

# ==== 실시간 그래프 세팅 ====
plt.ion()
fig, ax = plt.subplots()
window = deque(maxlen=50)
line, = ax.plot([], [], 'b-', lw=2)
ax.set_ylim(30, 100)
ax.set_xlim(0, 50)
ax.set_ylabel("dB Level")
ax.set_title("Real-time Floor Noise Level")

def update_plot(val):
    window.append(val)
    line.set_data(range(len(window)), list(window))
    plt.pause(0.01)

# ==== 메인 루프 ====
def loop():
    print("🎧 층간소음 감지 + 방향 분석 시작 (필터 포함)")
    with sd.InputStream(channels=2, samplerate=RATE, blocksize=CHUNK) as stream:
        while True:
            data, overflowed = stream.read(CHUNK)
            if overflowed:
                continue

            samples = np.frombuffer(data, dtype=np.int16)
            left = samples[::2]
            right = samples[1::2]

            # 필터 적용 (층간소음 대역만)
            left = bandpass_filter(left, RATE, LOWCUT, HIGHCUT)
            right = bandpass_filter(right, RATE, LOWCUT, HIGHCUT)

            dbL, dbR = rms_db(left), rms_db(right)
            snrL, snrR = snr_db(left), snr_db(right)
            avg_db = (dbL + dbR) / 2
            avg_snr = (snrL + snrR) / 2
            update_plot(avg_db)

            direction = "None"
            led_left.value = led_right.value = False

            if avg_db > THRESHOLD_DB and avg_snr >= SNR_MIN:
                delay = gcc_phat(left, right)
                if delay > 0:
                    direction = "LEFT"
                    led_left.value = True
                else:
                    direction = "RIGHT"
                    led_right.value = True

                msg = f"⚡ Shock Detected\nDir:{direction}\n{avg_db:.1f} dB"
                print(msg)
                show_text(msg)

                with open(CSV_LOG, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.strftime("%H:%M:%S"), dbL, dbR, snrL, snrR, direction])
            else:
                show_text("Quiet")

            time.sleep(0.1)

try:
    loop()
except KeyboardInterrupt:
    oled.fill(0)
    oled.show()
    print("종료됨.")