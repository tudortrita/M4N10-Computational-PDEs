%test code for tridiagonal matrix
% matrix is (N) x (N)
% a = diffusivity
% 

N=5;
% maxdt=11
% time=zeros(1,maxdt+1);
% u=zeros(N+1,maxdt+1);  % Array for u(x,t) initialised to zero
% u=zeros(N);
% u2=zeros(N);


% u(1,1)=0;
% u(N+1,1)=0;

A = sparse(N,N);
A = zeros(N,N);
A(1,1) = 2.0;
A(1,2) = -1.0;
RHS(1) = 1.0;
for i=2:N-1
  A(i,i-1) = -1.0;
  A(i,i) = 2.0;
  A(i,i+1) = -1.0;
  RHS(i) = 1.0;
end
A(N,N) = 2.0;
A(N,N-1) = -1.0;
B = A

RHS(N)= 1.0;

rhs = RHS

% tic and toc are start and end timers
  tic
  usolution = tridiag(A,RHS)
  toc
%
% u2=usolution
  

 

