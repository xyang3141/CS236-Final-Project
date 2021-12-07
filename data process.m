clear all; clc;
%params
namelist = dir('D:\学习\2021autumn\Research\DR_Train\LVR4313\accelerometer_data\channel1\normal\*.mat');
L = 1000;           % length of signal data interval
Fs = 1/L;           % Sampling frequency
T = 1/Fs;           % Sampling period
t = (0:L-1)*T;      % Time vector
%data generation
len = length(namelist);
rx = double.empty(L,0);
for i = 1:len
    i
    file_name = namelist(i).name;
    file = load(file_name);
    n = floor(length(file.save_var)/L);
    %rx = double.empty(m,0);
    for j = 0:n-1
        x = file.save_var(1+j*L:(j+1)*L);   %acceleration data
        y = fft(x);    %fast fourier transfer
        P2 = abs(y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        f = Fs*(0:(L/2))/L;
%         if i==1 & j==0
%             plot(f,P1);
%         end
        rx = [rx,P1];
        %rx = [rx,real(y)];
        %plot(rx);
        %save_name = ['D:\学习\2021autumn\Research\DR_Train\data generation\fft_',file_name(1:length(file_name)-4),'_',mat2str(j+1),'.csv'];
    end
%     save_name = ['D:\学习\2021autumn\Research\DR_Train\data generation\abnormal\fft_',file_name(1:length(file_name)-4),'.csv'];
%     csvwrite(save_name,rx);
end
%save_name = ['D:\学习\2021autumn\CS236\homework\completion\Project\final product\data\channel1\fft_',file_name(1:length(file_name)-8),'.csv'];
save_name = ['D:\学习\2021autumn\CS236\homework\completion\Project\final product\data\channel1\normal.csv'];
csvwrite(save_name,rx);

% x2 = save_var(1:10000); %acceleration data
% fx2 = fft(x2,1000); %fast fourier transfer
% rx2 = real(fx2); %get real part
% plot(rx2);


% csvwrite('D:\学习\2021autumn\Research\DR_Train\rx2.csv',rx2);