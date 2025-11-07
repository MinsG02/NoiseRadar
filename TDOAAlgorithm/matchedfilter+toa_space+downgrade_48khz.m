%% 환경 설정
c = 343; % 음속 [m/s]
fs = 48000; % (수정) 48kHz로 변경
t_total = 0:1/fs:0.05; % 50ms (fs=48k -> 2401 샘플)
sig_len = length(t_total); 
room_size = [2,2];
%% 'Chirp' 신호 생성 (40ms + 10ms Padding) (유지)
f0 = 100; % 100Hz
f1 = 4500; % 4500Hz
t_chirp_end = 0.04; % 40ms 
t_chirp = 0:1/fs:t_chirp_end; % (수정) t_chirp도 새 fs 적용
chirp_part = chirp(t_chirp, f0, t_chirp_end, f1)'; 
padding_len = sig_len - length(chirp_part);
chirp_signal = [chirp_part; zeros(padding_len, 1)];
%% 마이크 4개 '비대칭' 배치 (유지)
mic_pos = [0 0;  
           2 0;  
           0 2;  
           2 2]; 
num_mics = size(mic_pos,1); % 4
%% 발신 위치 예시 (랜덤 생성)
num_tx = 4; 
padding = 0.1; 
tx_pos = padding + (room_size(1) - 2*padding) * rand(num_tx, 2); 
disp('생성된 랜덤 발신 위치:');
disp(tx_pos);
%% FIR LPF (수신단 잡음 제거용)
lpFilt_rx = designfilt('lowpassfir','FilterOrder',64,'CutoffFrequency',4500,'SampleRate',fs); % (수정) 새 fs 적용
%% 수신 시뮬레이션
received = zeros(sig_len, num_mics, size(tx_pos,1));
disp('시뮬레이션 시작... (fs=48kHz, Noise=3.0, Matched Filter + Grid Search O)'); % (수정)
for k = 1:size(tx_pos,1)
    for m = 1:num_mics % m=1~4
        dist = norm(tx_pos(k,:) - mic_pos(m,:));
        delay_samp = dist/c * fs; % (수정) 새 fs 적용
        
        int_delay = floor(delay_samp);
        frac_delay = delay_samp - int_delay;
        temp_sig = [zeros(int_delay,1); chirp_signal(1:end-int_delay)]; 
        if frac_delay > 0
            temp_sig = (1-frac_delay)*temp_sig + frac_delay*[temp_sig(2:end); 0];
        end
        
        noise = 3*randn(sig_len,1); 
        
        received(:,m,k) = temp_sig + noise;
        received(:,m,k) = filter(lpFilt_rx, received(:,m,k)); 
    end
end
disp('시뮬레이션 완료.');
%% --- LSQNONLIN (Matched Filter + Coarse Grid Search 적용) ---
estimated_pos = zeros(size(tx_pos));
tdoa_pairs = [2 1; 3 1; 4 1;  
              3 2; 4 2;  
              4 3];     
num_tdoas = size(tdoa_pairs, 1); 
% 그리드 서치 설정 (반복문 바깥에서 한 번만 생성)
grid_step = 0.5; % 0.5m 간격
[X, Y] = meshgrid(0:grid_step:room_size(1), 0:grid_step:room_size(2));
grid_points = [X(:) Y(:)];
for k = 1:size(tx_pos,1)
    fprintf('TX 위치 %d/%d 계산 중...\n', k, size(tx_pos,1));
    
    % --- Matched Filter (xcorr) 기반 TOA 계산 ---
    toa_samp = zeros(num_mics, 1);
    for m = 1:num_mics
        [r, lags] = xcorr(received(:,m,k), chirp_signal);
        [~, I] = max(abs(r));
        toa_samp(m) = lags(I); 
    end
    % TOA의 차이로 TDOA 계산
    tdoa = zeros(num_tdoas, 1);
    for p = 1:num_tdoas
        mic_i = tdoa_pairs(p, 1); 
        mic_j = tdoa_pairs(p, 2);
        tdoa(p) = (toa_samp(mic_i) - toa_samp(mic_j)) / fs; % (수정) 새 fs 적용
    end
    % ---------------------------------------------------
    
    d_diff = tdoa * c; 
    
    fun = @(S) (sqrt((S(1)-mic_pos(tdoa_pairs(:,1),1)).^2 + (S(2)-mic_pos(tdoa_pairs(:,1),2)).^2) ... 
               - sqrt((S(1)-mic_pos(tdoa_pairs(:,2),1)).^2 + (S(2)-mic_pos(tdoa_pairs(:,2),2)).^2)) ...
               - d_diff; 
               
    % 1. '가벼운' 그리드 서치로 S0 찾기
    min_err = inf;
    best_S0 = [1.0, 1.0]; 
    
    for i = 1:size(grid_points, 1)
        err_vec = fun(grid_points(i, :)); 
        err = sum(err_vec.^2); 
        if err < min_err
            min_err = err;
            best_S0 = grid_points(i, :); 
        end
    end
    
    % 2. LSQNONLIN 정밀 최적화
    S0 = best_S0; 
    fprintf('  -> Coarse Grid Search 결과 S0 = [%.1f, %.1f]\n', S0(1), S0(2));
    
    options = optimoptions('lsqnonlin','Display','off');
    est = lsqnonlin(fun, S0, [0 0], [2 2], options); 
    
    estimated_pos(k,:) = est;
end
disp('계산 완료.');
%% --- 결과 출력 ---
disp('--- 4-Mic + 48kHz (Matched Filter + Grid Search, Noise=3.0) ---'); % (수정)
disp('실제 위치:'); disp(tx_pos);
disp('추정 위치:'); disp(estimated_pos);
error = sqrt(sum((tx_pos - estimated_pos).^2, 2));
disp('추정 오차 (평균):'); disp(mean(error));
%% 시각화: 신호 전후 비교
figure;
for k = 1:size(tx_pos,1)
    subplot(size(tx_pos,1),1,k);
    plot(t_total, chirp_signal,'b','DisplayName','Original Signal (Chirp)'); hold on;
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
title('TDOA 위치 추정 (4-Mic, 48kHz, Matched Filter + Grid Search, Noise=3.0)'); % (수정)
hold off;