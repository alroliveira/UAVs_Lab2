% crazyflie load usd card log csv file for sensor analysis
clear all
%close all
initPlots;

% available files:
csvfilename_motion = 'L2Data3.csv';

% read file
csvfilename = csvfilename_motion;
array = dlmread(csvfilename,',',1,0);

% get data from table
time = array(:,1)'*1e-3;
pos = array(:,2:4)'; % [m] groundtruth
vel = array(:,5:7)'; % [m/s] groundtruth
lbd = array(:,8:10)'; % [deg] grounclosedtruth
gyro = array(:,11:13)'; % [deg/s] sensor
acc = array(:,14:16)'; % [Gs] sensor
baro_asl = array(:,17)'; % [m] sensor
% lighthouse = array(:,18:20))'; % [m]

% convert date to print format
t = time - time(1);

% plot data
%initPlots;
%crazyflie_show_sensor_data(t,acc,gyro,baro_asl,pos,vel,lbd);

%% 1.4
g = 9.807;

theta_est  = (acc(1,:)/g)*180/pi;
phi_est = (acc(2,:)./ acc(3,:))*180/pi; 

figure;
plot(t, lbd(1,:)',t, phi_est(:));
legend('Real','Estimado');
title('$$\phi [deg]$$', 'interpreter','latex');

figure;
plot(t, lbd(2,:)',t, theta_est(:));
legend('Real','Estimado');
title('$$\theta [deg]$$', 'interpreter','latex');

%% 2.3 kalman filter

%x = [phi, w_phi, theta, w_theta, b_w_phi, b_w_theta]' 

A = [ 0 1 0 0 0 0 
      0 0 0 0 0 0
      0 0 0 1 0 0
      0 0 0 0 0 0
      0 0 0 0 0 0
      0 0 0 0 0 0 ];

y = [ acc(1,:)
      acc(2,:)
      acc(3,:)
      gyro(1,:)*pi/180
      gyro(2,:)*pi/180 ];

ny = 5;
nx = 6;

%

% kalman gains
escalar = 0.01;
R = 10^5*diag([1.3759e-06,1.2508e-06,1.6314e-06,0.0199,0.0323]); %covariancias obtidas no 1.1 (drone parado)
Q = escalar*eye(nx);

%Q = Ts^2*eye(nx);
%R = Ts^2*10*eye(ny);

C = [ 0 0 g 0 0 0
     -g 0 0 0 0 0
      0 0 0 0 0 0
      0 1 0 0 1 0 
      0 0 0 1 0 1 ];

rk_ctrb = rank(ctrb(A,Q));
rk_obsb = rank(obsv(A,C));

% initial values
P0 = 2*eye(nx);
x0 = zeros(nx,1);

% simulate system and estimation
Nsim = length(t);
Ts = mean(diff(t));
xe = zeros(nx,Nsim);
Pe = zeros(nx,nx,Nsim);
ye = zeros(ny,Nsim);
xe(:,1) = x0;
Pe(:,:,1) = P0;
F = expm(A*Ts);
Gu = 0;
H = C;
for k = 1:Nsim
    % predict next estimate:
    %xem = F*xe(:,k-1);
    %Pem = F*Pe(:,:,k-1)*F' + Q;    

    [xem,Pem] = kalman_predict(xe(:,k),Pe(:,:,k),F,Gu,Q);
    
    % update estimate with measurement info
    %K = Pem*C' * (C*Pem*C' + R)^-1;
    %xe(:,k) = xem + K * (y(:,k) - C*xem);
    %Pe(:,:,k) = (eye(nx) - K*C) * Pem;
    [xe(:,k),Pe(:,:,k),K] = kalman_update(xem,Pem,y(:,k),H,R);

    ye(:,k) = C*xe(:,k);
end


% Show results
figure;
subplot(2,1,1);
plot(t,xe(1,:)*180/pi);
hold on;
plot(t,lbd(1,:),"LineWidth", 1);
grid on;
xlabel('t [s]');
ylabel('$$\phi$$ [deg]');
legend('est','real');
title('roll angle estimates')

subplot(2,1,2);
plot(t,xe(3,:)*180/pi);
hold on;
plot(t,lbd(2,:),"LineWidth", 1);
grid on;
xlabel('t [s]');
ylabel('$$\theta$$ [deg]');
legend('est','real');
title('pitch angle estimates')

figure;
subplot(2,1,1);
plot(t,xe(2,:)*180/pi);
hold on;
plot(t,gyro(1,:),"LineWidth", 1);
grid on;
xlabel('t [s]');
ylabel('$$\omega_\phi$$ [deg/s]');
legend('est','real');
title('angular velocity $$\omega_x$$ estimates')

subplot(2,1,2);
plot(t,xe(4,:)*180/pi);
hold on;
plot(t,gyro(2,:),"LineWidth", 1);
grid on;
xlabel('t [s]');
ylabel('$$\omega_\theta$$ [deg/s]');
legend('est','real');
title('angular velocity $$\omega_y$$ estimates')

% figure;
% plot(t,xe(5,:),t,xe(6,:));
% grid on;
% xlabel('t [s]');
% legend('$$b w_\phi$$','$$b w_\theta$$');

%% 3.3 extended kalman filter

% initial values
P0 = 2*eye(nx);
x0 = zeros(nx,1);

% simulate system and estimation
Nsim = length(t);
Ts = mean(diff(t));
xeEKF = zeros(nx,Nsim);
Pe = zeros(nx,nx,Nsim);
ye = zeros(ny,Nsim);
xeEKF(:,1) = x0;
Pe(:,:,1) = P0;
F = expm(A*Ts);
Gu = 0;

for k = 2:Nsim
    % predict next estimate:
    xem = F*xeEKF(:,k-1) + Gu;
    Pem = F*Pe(:,:,k-1)*F' + Q;    
    Pem = (Pem+Pem')/2;

    Ci = [ 0                                    0 g*cos(xeEKF(3,k))                 0 0 0
          -g*cos(xeEKF(1,k))*cos(xeEKF(3,k))    0 g*sin(xeEKF(1,k))*sin(xeEKF(3,k)) 0 0 0
           g*sin(xeEKF(1,k))*cos(xeEKF(3,k))    0 g*cos(xeEKF(1,k))*sin(xeEKF(3,k)) 0 0 0
           0                                    1 0                                 0 1 0
           0                                    0 0                                 1 0 1 ];
    H = Ci;
    
    % update estimate with measurement info
    K = Pem*H' / (H*Pem*H' + R);
    xeEKF(:,k) = xem + K * (y(:,k) - H*xem);
    Pe(:,:,k) = (eye(nx) - K*H) * Pem;
    Pe(:,:,k) = (Pe(:,:,k) + Pe(:,:,k)') / 2;

    ye(:,k) = C*xeEKF(:,k);
end

% Show results
figure;
subplot(2,1,1);
plot(t,xeEKF(1,:)*180/pi,t,xe(1,:)*180/pi);
hold on;
plot(t,lbd(1,:),"LineWidth", 1);
grid on;
xlabel('t [s]');
ylabel('$$\phi$$ [deg]');
legend('EKF','KF','real');
title('roll angle estimates')

subplot(2,1,2);
plot(t,xeEKF(3,:)*180/pi,t,xe(3,:)*180/pi);
hold on;
plot(t,lbd(2,:),"LineWidth", 1);
grid on;
xlabel('t [s]');
ylabel('$$\theta$$ [deg]');
legend('EKF','KF','real');
title('pitch angle estimates')

figure;
subplot(2,1,1);
plot(t,xeEKF(2,:)*180/pi,t,xe(2,:)*180/pi);
hold on;
plot(t,gyro(1,:),"LineWidth", 1);
grid on;
xlabel('t [s]');
ylabel('$$\omega_\phi$$ [deg/s]');
legend('EKF','KF','real');
title('angular velocity $$\omega_x$$ estimates')

subplot(2,1,2);
plot(t,xeEKF(4,:)*180/pi,t,xe(4,:)*180/pi);
hold on;
plot(t,gyro(2,:),"LineWidth", 1);
grid on;
xlabel('t [s]');
ylabel('$$\omega_\theta$$ [deg/s]');
legend('EKF','KF','real');
title('angular velocity $$\omega_y$$ estimates')
