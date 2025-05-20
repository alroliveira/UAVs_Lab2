% crazyflie load usd card log csv file

% read file
csvfilename = '2024-04-04_log07.csv';

array = dlmread(csvfilename,',',1,0);
%T = table2array(readtable(csvfilename)); % Matlab only

% get data from table (octave)
time = array(:,1)'*1e-3;
pos = array(:,2:4)';
vel = array(:,5:7)';
lbd = array(:,8:10)'*pi/180;
om = array(:,11:13)'*pi/180;
pos_ref = array(:,14:16)';
yaw_ref = array(:,17)';
motors = array(:,18:21)';

% convert date to print format
t = time - time(1);
x = [pos;vel;lbd;om];
x_ref = [pos_ref;0*vel;lbd*0;om*0];
x_ref(9,:) = yaw_ref;
uint16_max = 2^16;
u = motors/uint16_max;

% plot data
initPlots;
vehicle3d_ref_show_data(t,x,u,x_ref);

% prepare data for ID
% dt = diff(t);
% dt = [dt,dt(end)];
% sample_time_stats = [mean(dt),min(dt),max(dt)],
% Ts = sample_time_stats(1);
u_id = lbd(1,:)';
y_id = pos(2,:)';
