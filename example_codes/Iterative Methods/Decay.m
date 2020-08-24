% How does the error behave during iterative schemes?
% Elliptic PDE, with f=0, mestel-> mughal (some modifications Feb14-2020
% Jac calls teh Jacobi iteraion method; GS the Gauss-Seidel; SOR as G-S
% with Relaxation parameter omega set at 1.3
% EACH CALL, i.e : Jac(u,b,2), the 2 means return the iterated solution
% after 2 global iteration sweeps across the computaional domain.
%
clear
N=21;h=1/(N-1);
for m=1:N
    x(m)=(m-1)*h;
    for n=1:N
        y(n)=(n-1)*h;
        b(m,n)=0; % Exact solution is u=0 everywhere
        uinit(m,n)=x(m)*(1-x(m))*y(n)*(1-y(n))*(2*sin(10*x(m)*pi)+3*sin(15*y(n)*pi)^2);
        %uinit(m,n)=sin(3*x(m)*pi)*sin(3*y(n)*pi)+.6*x(m)*y(n);
        %uinit(m,n)=(x(m)-0.5)+3*(y(n)-0.4);
    end
end
for mn=1:N
    uinit(1,mn)=0;
    uinit(N,mn)=0;
    uinit(mn,1)=0;
    uinit(mn,N)=0;
end
u=uinit;
az=-76; 
el=60;
figure(1)
surf(uinit); 
caxis([0. 0.3]);
colorbar;
   view([az, el]);
   shading interp
colormap('jet');
for i=1:41
%   u=GS(u,b,2);
   u=Jac(u,b,2);
%   u=SOR(u,b,1.3,2);
   figure(2)
%   contour(u);
   surf(u);
   caxis([0. 0.1]);
   colormap('jet');
   view([az, el]);
   shading interp
   colorbar
   pause(1)
end


