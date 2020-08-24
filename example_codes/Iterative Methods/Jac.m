function u = Jac(uinit,b,Niter)

%------------------------------------------%
% Takes Niter Jacobi iterations to         %
% solve the Poisson equation in form Au=b  %
% on unit square starting from uinit       %
%------------------------------------------%

  % resolution
  m = size(b,1);
  n = size(b,2);
  %dx = 1/(m-1);
  %dy = 1/(n-1);
  m2 = (m-1)*(m-1);
  n2 = (n-1)*(n-1);
  m2n22 = 2*(m2 + n2); 

  % initialization
  uold = uinit;
  unew=uold;
  
  % iteration
  for k=1:Niter
    for i=2:m-1
      for j=2:n-1
	unew(i,j) = (m2*(uold(i+1,j)+uold(i-1,j)) + ...
	             n2*(uold(i,j+1)+uold(i,j-1)) - ...
	             b(i,j))/m2n22;
      end
    end
  % For Jacobi, find all the new iterates before updating u
  % Store two arrays
    uold = unew;
  end

  u = unew;
  