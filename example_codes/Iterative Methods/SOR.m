
function unew = SOR(uinit,b,omega,Niter)

%------------------------------------------%
% Takes Niter Gauss-Seidel iterations      %
% relaxing with parameter omega >1  to     %
% solve the Poisson equation in form Au=b  %
% on unit square starting from uinit       %
%------------------------------------------%
  
  m = size(b,1);
  n = size(b,2);
  %dx = 1/(m-1);
  %dy = 1/(n-1);
  m2 = (m-1)*(m-1);
  n2 = (n-1)*(n-1);
  m2n22 = 2*(m2 + n2); 

  % initialization
  unew = uinit;
  
  % iteration
  for k=1:Niter
    for i=2:n-1
      for j=2:m-1
          % For Gauss-Seidel, overwrite u and use calculated new values
	      utem = (m2*(unew(i+1,j)+unew(i-1,j)) + ...
	          n2*(unew(i,j+1)+unew(i,j-1)) - ...
	          b(i,j))/m2n22;
          % Now relax. omega>1 for over-relaxation
          unew(i,j)=(1-omega)*unew(i,j)+omega*utem;
      end
    end
  end

  u = unew;
  