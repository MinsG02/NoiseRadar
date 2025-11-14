#2mic tdoa 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INMP441 x2 (I2S) 실시간 스테레오 모니터
- 2채널 RMS(dBFS) 비교로 LEFT/RIGHT/CENTER 판정
- 콘솔 실시간 출력
- (옵션) I2C LCD에 상태 표시: RPLCD가 설치되어 있으면 자동 사용
"""

import numpy as np
import sounddevice as sd
import time
import math

# ===== 사용자 설정 =====
SAMPLE_RATE = 16000      # 16 kHz (필요시 48k로 올려도 됨)
BLOCK_SIZE  = 1024       # 블록 크기(프레임 수)
DEVICE = None            # None이면 기본 입력장치. 예: "hw:1,0" 또는 장치명 문자열
DB_DIFF = 3.0            # L/R 우세 판정 임계(dB). 3dB 이상 차이면 우세로 간주
PRINT_EVERY = 0.05       # 출력 갱신 주기(초)

# ===== LCD (옵션) =====
USE_LCD = True
LCD_ADDR = 0x27          # i2cdetect로 확인된 주소
LCD_EXPANDER = 'PCF8574' # 0x38~0x3F 대역이면 'PCF8574A'

lcd = None
if USE_LCD:
    try:
        from RPLCD.i2c import CharLCD
        lcd = CharLCD(
            i2c_expander=LCD_EXPANDER,
            address=LCD_ADDR,
            port=1,
            cols=16, rows=2,
            charmap='A02',
            auto_linebreaks=True
        )
        lcd.clear()
        lcd.write_string('Stereo Monitor')
        lcd.cursor_pos = (1, 0)
        lcd.write_string('Starting...')
        time.sleep(0.5)
        lcd.clear()
    except Exception as e:
        print(f'[LCD] 사용 안 함: {e}')
        lcd = None

# ===== 유틸 =====
INT32_FS = float(np.iinfo(np.int32).max)

def rms_dbfs(x: np.ndarray) -> float:
    # x: int32 파형
    if x.size == 0:
        return -120.0
    # 오버플로 방지 위해 float64 변환
    x = x.astype(np.float64)
    rms = math.sqrt(np.mean(x * x) + 1e-20)
    db = 20.0 * math.log10(rms / INT32_FS + 1e-20)
    return db

def classify(db_l: float, db_r: float, diff_db: float = DB_DIFF) -> str:
    d = db_l - db_r
    if d > diff_db:
        return 'LEFT '
    elif d < -diff_db:
        return 'RIGHT'
    else:
        return 'CENTER'

# ===== 콜백 방식 스트리밍 =====
last_print = 0.0

def audio_callback(indata, frames, time_info, status):
    global last_print
    if status:
        print('[ALSA]', status)

    # indata: shape (frames, channels)
    if indata.shape[1] < 2:
        print('⚠ 입력이 모노입니다. I2S/디바이스 설정을 확인하세요.')
        return

    # int32 캡처 가정(S32_LE). 다른 dtype이면 변환
    if indata.dtype != np.int32:
        data = indata.copy().view(np.int32) if indata.dtype == np.int32 else indata.astype(np.int32)
    else:
        data = indata

    left  = data[:, 0]
    right = data[:, 1]

    dbL = rms_dbfs(left)
    dbR = rms_dbfs(right)
    tag = classify(dbL, dbR, DB_DIFF)

    now = time.time()
    if now - last_print >= PRINT_EVERY:
        barL = int(np.clip((dbL + 60) / 3, 0, 20)) * '■'
        barR = int(np.clip((dbR + 60) / 3, 0, 20)) * '■'
        print(f'{tag} | L {dbL:6.1f} dBFS {barL:<20} | R {dbR:6.1f} dBFS {barR:<20}', end='\r')
        last_print = now

        if lcd:
            try:
                lcd.cursor_pos = (0, 0)
                lcd.write_string('L:{:5.1f} R:{:5.1f} '.format(dbL, dbR).ljust(16)[:16])
                lcd.cursor_pos = (1, 0)
                lcd.write_string(('-> ' + tag).ljust(16)[:16])
            except Exception as e:
                print(f'\n[LCD ERROR] {e}')

def main():
    print('Starting stereo monitor...')
    print('- 장치:', DEVICE if DEVICE else '(기본 장치)')
    with sd.InputStream(device=DEVICE,
                        channels=2,
                        samplerate=SAMPLE_RATE,
                        dtype='int32',       # ALSA: S32_LE
                        blocksize=BLOCK_SIZE,
                        callback=audio_callback):
        print('Press Ctrl+C to stop.')
        while True:
            time.sleep(0.1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
    except Exception as e:
        print(f'\n[ERROR] {e}\n'
              f'- arecord로 2채널 캡처 가능한지 먼저 확인\n'
              f'- 장치 이름/번호가 다르면 DEVICE 변수 설정 (예: DEVICE=\"hw:1,0\")\n'
              f'- dtoverlay와 I2S 배선(BCLK, LRCLK, DIN, 3V3/5V, GND) 점검')
