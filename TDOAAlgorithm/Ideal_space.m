%% 환경 설정
c = 343; % 음속 [m/s]
fs = 192000; % (유지) 192kHz
t_total = 0:1/fs:0.05; % 50ms
sig_len = length(t_total); % 9601 샘플
room_size = [2,2];

%% 'Chirp' 신호 생성 (40ms + 10ms Padding) (유지)
f0 = 100; % 100Hz
f1 = 4500; % 4500Hz
t_chirp_end = 0.04; % 40ms 
t_chirp = 0:1/fs:t_chirp_end;
chirp_part = chirp(t_chirp, f0, t_chirp_end, f1)'; 
padding_len = sig_len - length(chirp_part);
chirp_signal = [chirp_part; zeros(padding_len, 1)];

%% 마이크 4개 '비대칭' 배치 (유지)
mic_pos = [0.1 0.1;  
           1.9 0.2;  
           0.2 1.9;  
           1.8 1.8]; 
num_mics = size(mic_pos,1); % 4

%% 발신 위치 예시
tx_pos = [0.5 0.5; 0.5 1.5; 1.5 0.5; 1.5 1.5];

%% FIR LPF (수신단 잡음 제거용)
lpFilt_rx = designfilt('lowpassfir','FilterOrder',64,'CutoffFrequency',4500,'SampleRate',fs);

%% 수신 시뮬레이션
received = zeros(sig_len, num_mics, size(tx_pos,1));
disp('시뮬레이션 시작... (fs=192kHz, Noise=0)');
for k = 1:size(tx_pos,1)
    for m = 1:num_mics % m=1~4
        dist = norm(tx_pos(k,:) - mic_pos(m,:));
        delay_samp = dist/c * fs;
        
        % (Fractional Delay 선형 보간법 유지)
        int_delay = floor(delay_samp);
        frac_delay = delay_samp - int_delay;
        temp_sig = [zeros(int_delay,1); chirp_signal(1:end-int_delay)]; 
        if frac_delay > 0
            temp_sig = (1-frac_delay)*temp_sig + frac_delay*[temp_sig(2:end); 0];
        end
        
        % **(수정 1) 잡음 0으로 설정**
        noise = 0; % 0.01*randn(sig_len,1); 
        
        received(:,m,k) = temp_sig + noise;
        received(:,m,k) = filter(lpFilt_rx, received(:,m,k)); 
    end
end
disp('시뮬레이션 완료.');

%% --- Grid Search + LSQNONLIN ---

%% GCC-PHAT 기반 TDOA 계산
estimated_pos = zeros(size(tx_pos));
tdoa_pairs = [2 1; 3 1; 4 1;  
              3 2; 4 2;  
              4 3];     
num_tdoas = size(tdoa_pairs, 1); 

N_fft = 2*sig_len; % 2배로 패딩 (유지)

disp('Grid Search 시작... (10cm 해상도, 즉시 완료)');
for k = 1:size(tx_pos,1)
    fprintf('TX 위치 %d/%d 계산 중...\n', k, size(tx_pos,1));
    tdoa = zeros(num_tdoas, 1);
    for p = 1:num_tdoas
        mic_i = tdoa_pairs(p, 1); 
        mic_j = tdoa_pairs(p, 2);
        
        X1 = fft(received(:,mic_i,k), N_fft);
        X2 = fft(received(:,mic_j,k), N_fft); 
        R = X1.*conj(X2)./(abs(X1.*conj(X2)) + eps); 
        r = ifft(R);
        
        [~,I] = max(abs(r));
        lag = I-1;
        if lag > N_fft / 2 
            lag = lag - N_fft;
        end
        tdoa(p) = lag/fs; 
    end
    
    d_diff = tdoa * c; 
    
    fun = @(S) (sqrt((S(1)-mic_pos(tdoa_pairs(:,1),1)).^2 + (S(2)-mic_pos(tdoa_pairs(:,1),2)).^2) ... 
               - sqrt((S(1)-mic_pos(tdoa_pairs(:,2),1)).^2 + (S(2)-mic_pos(tdoa_pairs(:,2),2)).^2)) ...
               - d_diff; 
               
    % **(수정 2) Grid Search (10cm 해상도)**
    grid_step = 0.1; % 10cm (21x21 = 441 지점)
    [X, Y] = meshgrid(0:grid_step:2, 0:grid_step:2); 
    
    sse_map = zeros(size(X));
    for i = 1:size(X,1)
        for j = 1:size(X,2)
            sse_map(i,j) = sum(fun([X(i,j), Y(i,j)]).^2);
        end
    end
    
    % 2. 최소 SSE 지점(초기값) 찾기
    [~, min_idx] = min(sse_map(:)); 
    [row, col] = ind2sub(size(sse_map), min_idx); 
    S0_grid = [X(row, col), Y(row, col)];
    
    % 3. LSQNONLIN 정밀 최적화
    options = optimoptions('lsqnonlin','Display','off');
    est = lsqnonlin(fun, S0_grid, [0 0], [2 2], options); 
    
    estimated_pos(k,:) = est;
end
disp('Grid Search 완료.');

%% --- 결과 출력 ---
disp('--- 4-Mic + 192kHz + 10cm Grid (Noise=0) ---');
disp('실제 위치:'); disp(tx_pos);
disp('추정 위치:'); disp(estimated_pos);
error = sqrt(sum((tx_pos - estimated_pos).^2, 2));
disp('추정 오차 (평균):'); disp(mean(error));

%% 시각화: 신호 전후 비교
figure;
for k = 1:size(tx_pos,1)
    subplot(size(tx_pos,1),1,k);
    plot(t_total, chirp_signal,'b','DisplayName','Original Signal (Chirp+Pad)'); hold on;
    plot(t_total, received(:,1,k),'r','DisplayName','Filtered Mic1'); 
    xlabel('Time [s]'); ylabel('Amplitude'); grid on;
    legend; 
    title(['TX Signal ', num2str(k),' - Original vs Filtered']);
end

%% 위치 시각화
figure; hold on;
plot(mic_pos(:,1),mic_pos(:,2),'ro','MarkerSize',12,'DisplayName','Mic'); 
plot(tx_pos(:,1),tx_pos(:,2),'b*','MarkerSize',10,'DisplayName','True TX');
plot(estimated_pos(:,1),estimated_pos(:,2),'kx','MarkerSize',10,'DisplayName','Estimated TX');
legend; 
xlim([0 2]); ylim([0 2]); 
grid on;
xlabel('X [m]'); ylabel('Y [m]');
title('TDOA 위치 추정 (4-Mic, 192kHz, 10cm Grid, Noise=0)'); 
hold off;
Update test;
