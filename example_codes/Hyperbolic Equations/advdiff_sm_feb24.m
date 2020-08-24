% Explicit method to solve the advection diffusion equation u_t + u_x = 0
% Parameters: 0 < x < L, N=number of points
% a = diffusivity, V(x)=velocity
%
clear
L=1;
% >> NOT USED>>>>
a=0.4;
Vmax=1.;
% DX/DT=1 >>>>
N=100;  
maxdt=101; % Maximum number of timesteps
dt=0.01; % timestep k
dx=L/N; % or h

dtq=dx*dx; 
h=dx;


% >> NOT USED>>>> V too!
r=a*dt/dx^2; 
q=Vmax*dx/a;

rk=dt/h;
rksq=rk*rk;

u=zeros(N+1,maxdt+1);  % Array for u(x,t) initialised to zero
x=zeros(1,N+1);
V=zeros(1,N+1);
time=zeros(1,maxdt+1);
for n=1:(N+1)        % Set up grid x(n) and initialise u(n,1)
    x(n)=(n-1)*dx;
    V(n)=Vmax;
    if(abs(x(n))<=0.5)     
        u(n,1)=1;
    else
        u(n,1)=0;
    end
   % u(n,1)=exp(-x(n)^2);
end
u(1,1)=1; % Zero boundary conditions
u(N+1,1)=0;
uu(1)=u(1,1);
uu(N+1)=0;

time(1)=0;
hold off
for j=1:maxdt    % Step in time
    time(j+1)=time(j)+dt;
    for n=2:N
% centered >>  (1) no good
 %      u(n,j+1)=u(n,j)+dt*((u(n-1,j)-u(n+1,j)))/(2*dx);
% bwinded >>  (2) no good
%        u(n,j+1)=u(n,j)+dt*((u(n,j)-u(n-1,j)))/(dx);
% upwinded  >> (3) good
%         u(n,j+1)=u(n,j)+dt*((u(n-1,j)-u(n,j)))/(dx);

% lax-wendroff:  good
	u(n,j+1)=0.5*(rksq-rk)*u(n+1,j)+0.5d0*(rksq+rk)*u(n-1,j)+u(n,j)*(1.d0-rksq);


        uu(n)=u(n,j+1)
    end;
    u(1,j+1)=1;
    u(N+1,j+1)=0;
figure(1);    % Contour u(x,t)
plot(x,uu);
axis([0. 1. -0.2 1.2]);
% hold off
 hold on
 pause(1)
end;
