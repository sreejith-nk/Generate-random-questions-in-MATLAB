%Linerar Algebra
fprintf("\nLINEAR ALGEBRA\n\n")
for i=1:5
    A=randi([1,9],3,3);
    sym=A+A';
    
    %pos_d will be positive definite because sym is symmetric and have real eigen values. So eigen values of pos_d will be square of real values. 
    pos_d=sym*sym;
    
    %pos_d will be negative definite because sym is symmetric and have real eigen values. So eigen values of neg_d will be minus of square of real values. 
    neg_d=-sym*sym;
    
    r=rand();
    fprintf("\nQ 1, V %d\n",i);
    [V,D]=eig(pos_d);
    B=V*sqrt(D);
    B*B';
    D=rand(3,3)*40;

    if r>0.5
        fprintf("Find matrix B with real entries such that X=B*B^T using Eigen Value Decomposition.\n ")
        fprintf("X=\n%d\t%d\t%d;\n%d\t%d\t%d;\n%d\t%d\t%d;\n",pos_d)
        fprintf("\nA.B matrix does not exist\nB.B matrix exists but cannot be calculated using given data\nC.B=[%d\t%d\t%d;%d\t%d\t%d;%d\t%d\t%d]\nD.B=[%d\t%d\t%d;%d\t%d\t%d;%d\t%d\t%d]\nE.None of these\n",B',D)
        fprintf("\nANSWER: C\n")
        fprintf("SOLUTION\n")
        fprintf("Eigen values of X are [%d;%d;%d]\n",eig(pos_d))
        fprintf("Given matrix is symmetric and has positive eigen values. So it is a positive definite matrix.\nSo we know,  X=V*D*V^T\tOR\tX=(V*D^0.5)*(D^0.5*V^T)(Since eigen values are positive, we can take root of diagonal matrix\n Now this is of the form B*B^T\n So B=V*D^0.5\n")
        B
    else
        fprintf("Find matrix B with real entries such that X=B*B^T using eigen value decomposition.\n ")
        fprintf("X=\n%d\t%d\t%d;\n%d\t%d\t%d;\n%d\t%d\t%d;\n",neg_d)
        fprintf("\nA.B matrix does not exist\nB.B matrix exists but cannot be calculated using given data\nC.B=[%d\t%d\t%d;%d\t%d\t%d;%d\t%d\t%d]\nD.B=[%d\t%d\t%d;%d\t%d\t%d;%d\t%d\t%d]\nE.None of these\n",B',D)
        fprintf("\nANSWER: A")
        fprintf("\nSOLUTION\n")
        fprintf("Eigen values of X are [%d;%d;%d]\n",eig(neg_d))
        fprintf("Given matrix is symmetric and has negative eigen values. So it is a negative definite matrix.\nSo we know,  X=V*D*V^T\tOR\tX=(V*D^0.5)*(D^0.5*V^T)(Since eigen values are negative, we can't take root of diagonal matrix.)\nSo no real B exists.\n")
    end
end

for i = 1:5
    fprintf('\n Q 2 , V %d .\n \n',i);
    a =1+randi(10);
    k1 = sqrt(a);
    k2 = -sqrt(a);
    r1 = 4+randi(10);
    A1 = [k1 a 0 3;-1 0 k1 2;0 1 1 1];
    A2 = [k2 a 0 3;-1 0 k2 2;0 1 1 1];
    A1reduced = rref(A1);
    A2reduced = rref(A2);
    Q = ['consider the matrix A = [k a 0;-1 0 k;0 1 1] ,where k is a real number and a = ',num2str(a) ,', x = [x1;x2;x3] B = [3;2;1]' ,'and  Equation E = Ax=B .',' Identify the wrong statements below'];
    disp(Q)
    option1 = ['A)For k = + or - ', num2str(sqrt(a)),'. The matrix A is non-invertible'];
    option2 = ['B)For k = ',num2str(r1), ', The system of equation has exaclty one solution' ];
    option3 = ['C)For k = + or - ', num2str(sqrt(a)),', The system of equation has infinetly many solution'];
    option4 = ['D)For k = + or - ', num2str(sqrt(a)),', The system of equation has no solutions'];
    
    disp(option1)
    disp(option2)
    disp(option3)
    disp(option4)
   
    explainationA = ['Expand the determinant,The matrix is singular if k^2 =',num2str(a),'or k = + or -',num2str(sqrt(a)),', then the matrix is non invertible' ];
    explainationB = ['For any k not equal to + or -',num2str(sqrt(a)), ', A is invertible and system has a unique solution' ];
    explainationC = ['For k equal to + or - ',num2str(sqrt(a)),', The corresponding Augmented matrix are given below'];
    explaination = ['so the system of equation will not have infinitely many solutions'];
    
    fprintf('\n (C) is the wrong statement! \n')
    disp(explainationA)
    disp(explainationB)
    disp(explainationC)
    A1
    A2
    
    A1reduced
    A2reduced
    disp(explaination)
end



%Optimiziation

fprintf("\n\nOPTIMIZATION\n")
for i = 1:5
    
    global Area
    Area = randi(10);
    
    %set initial guess values for the box dimension
    lengthguess = 1;
    widthguess = 1;
    heightguess = 1;
    %load guess values into array
    x0 = [lengthguess widthguess heightguess];
    
    %call solver to minimise 
    xopt = fmincon(@objective,x0,[],[],[],[],[],[],@constraint,[]);
    
    %retrive optimized box size and volume 
    volumeOpt = calcVolume(xopt);
    
    %calculate surface area with optimal solution
    surfaceAreaOpt = calcSurface(xopt);
    
    fprintf('Ignor the message above , it comes from running the optimization code! \n');
    fprintf('Q 3 , V %d \n',i);
    
    Q = ['Consider a box, maximise the volume of the box by adjusting the length, width and height subjected to the constraint that the surface area is less than or equal to ',' ', num2str(Area)] ;
    %display question
    disp(Q)
    
    option1 = num2str(10*rand(1,1));
    option2 = num2str(10*rand(1,1));
    option3 = num2str(volumeOpt);
    option4 = num2str(10*rand(1,1));
    
    %display options
    fprintf('A ) %s \n',option1);
    fprintf('B ) %s \n',option2);
    fprintf('C ) %s \n',option3);
    fprintf('D ) %s \n',option4);
    
    solution = ['the correct answer is option C'];
    disp(solution)
    
    volumeOpt = calcVolume(xopt)
    surfaceAreaOpt = calcSurface(xopt)
    xopt    
end
for i=1:5
syms x y z p q lambda mu;
a=randi([1,5]);
b=randi([1,4]);
c=randi([1,5]);
d=randi([1,4]);
e=randi([1,4]);
z=(x-a)^2 + b*y -c;
p=x+d*y-2;
q=e*y-x-1;
%LAGRANGIAN FUNCTION
l=z+lambda*q+mu*p;
fprintf("Q 4 V %d\n",i)
%QUESTION
fprintf("For what values of x and y the function %s=0 minimises \nsubject to the constraints  %s<=0  and  %s=0\nSolve using KKT conditions\n",char(z),char(p),char(q));
%DIFFERENTIATING LAGRANGIAN
r=diff(l,x);
s=diff(l,y);
%SOLVING THE KKT CONDITIONS
eqns = [r==0,s==0,p==0,q==0];
S = solve(eqns,[mu lambda x y]);
mu_ans1=S.mu;
lambda_ans1=S.lambda;
x_ans1=S.x;
y_ans1=S.y;
%CHECKING INEQUALITY AND VALUE OF mu
if (mu_ans1 <= 0 || x_ans1+d*y_ans1 - 2> 0)
    eqns = [r==0,s==0,q==0,mu==0];
      S = solve(eqns,[mu lambda x y]);
      x=S.x;
      y=S.y;
      mu=S.mu;
      lambda=S.lambda;
else
    x=x_ans1;
    y=y_ans1;
    mu=mu_ans1;
    lambda=lambda_ans1;
    
end
fprintf('A)x=1/%d,y=2/%d\n',randi([1,8]),randi([5,28]))
fprintf('B)x=1/%d,y=1/%d\n',randi([4,8]),randi([3,28]))
fprintf('C)x=1/%d,y=3/%d\n',randi([7,16]),randi([9,28]))
fprintf('D)x=%s,y=%s\n\n\n',x,y)
%SOLUTION
fprintf("solution\n");
fprintf('D) is the correct answer.\n');
fprintf('Lagrangian\n%s\n',l);
fprintf('First order KKT conditions\n%s=0\n%s=0\nlambda*(%s)=0\nmu*(%s)=0\n',r,s,q,p);
fprintf('First we will solve by assuming mu>=0.\nSolving we get x=%s,y=%s,mu=%s,lambda=%s\n',x_ans1,y_ans1,mu_ans1,lambda_ans1);
fprintf('Check whether the inequality is satisfied and mu is greater than zero\n');
if (mu_ans1 <= 0 || x_ans1+d*y_ans1 - 2> 0)
    fprintf('Our assumption is wrong.So we will solve by using mu=0\n');
    fprintf('Solving we get x=%s,y=%s,mu=%s,lambda=%s\n',x,y,mu,lambda');
    fprintf('So the answer is x=%s,y=%s\n\n\n\n',x,y);
else
    fprintf('Our assumption is right.\nSo the answer is x=%s,y=%s\n\n\n\n',x,y);
    
end
end  






%Probability and statistics
fprintf("\nPROBABILITY AND STATISTICS\n")
for i=1:5
r= randi(100,1,10);
a= randi([1,10]);b= randi([1,10]);c= randi([1,10]);
m1=mean(r);
m2=((rssq(r))^2)/10;
z=0;
for j=1:10
z=z+(r(j))^3;
end
m3=z/10;
x=a*m1 + b*m2 + c*m3;
fprintf("\n");
fprintf("\nQ 5 , V%d\n",i);
fprintf("Consider the numbers %d, %d, %d, %d, %d, %d, %d, %d, %d, %d. Let M1, M2, M3 be the first 3 moments of this data set. Find %d*M1 + %d*M2 + %d*M3. \n", r(1), r(2), r(3), r(4), r(5), r(6), r(7), r(8), r(9), r(10),a,b,c)
fprintf("A. %d\n B. %d\n C. %d\n D. %d\n E.None of these\n",x+1894,x,x-2000,x+3567);
fprintf("Answer: B\n");
fprintf("\nSOLUTION\n");
fprintf("M1 is the mean.\n M1=(sum of all the numbers)/10 = %d\n",m1);
fprintf("M2 is the second moment.\n M2=(Sigma (Xi^2))/10 = %d\n",m2);
fprintf("M3 is the third moment.\n M3=(Sigma (Xi^3))/10 = %d\n",m3);
fprintf("%d*M1 + %d*M2 + %d*M3 = %d\n",a,b,c,x);
end

for i=1:5    
    n=randi([10,50]);
    y=randi([300,700]);
    p_1=1;
    fprintf("\n")
    fprintf("\nQ 6,V %d\n",i)
    fprintf("In a room of %d peopele. What is the probability that atleast 2 people have same birthdays? Their planet has %d days in a year\n",n,y)
    
    %code to calculate the multiplication series to obtain probability
    for i=y+1-n:y
        p_1=p_1*i/y;
    end
    
    %QUESTION
    p_1;
    p_2=1-p_1;
    fprintf("A.2/%d\nB.%d/%d\nC.%d\nD.%d\nE.none of these\n",n,n,y,p_1,p_2)
    
    %SOLUTION
    fprintf("\nSOLUTION")
    fprintf("\nANSWER=D\n")
    fprintf("We will calculate the probability that no 2 people have same birthday and subtract it from one to get our answer\n")
    fprintf("P(No 2 same birthdays in %d people) = %d*...(%d-(%d-1)/%d^(%d-1) \n",n,y,y,n,y,n)
    fprintf("P(No 2 same birthdays in %d people) = %d\n",n,p_1)
    fprintf("probability that atleast 2 people have same birthday = %d",p_2)
end



%functions used in Question 3 optimization is given below
%Define Funstion to calculate vomlume of a box
function volume = calcVolume(x)
    length = x(1);
    width = x(2);
    height = x(3);
    volume = length * width * height; 
end

%Define a functionn to calculate the surface area of box
function surfaceArea = calcSurface(x)
    length = x(1);
    width = x(2);
    height = x(3);
    surfaceArea = 2*length*width + 2*length*height + 2*height*width;  
end

% Define objective function for optimization
function obj = objective(x)
    obj = -calcVolume(x);
end

% Define constraints for optimization
function [c, ceq] =constraint(x)
    global Area
    c = calcSurface(x) - Area;
    ceq = [];
end