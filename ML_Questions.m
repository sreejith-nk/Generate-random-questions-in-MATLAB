%Question 1
for i=1:5    
    fprintf("Q 1, V %d\n",i)
    fprintf("Given the follwing information from 4 different spam classifiers, Identify the most suitable algorithm among them, If more than one is equally good mark the option with lowest Algorithm number. That means if algorithms 1 and 2 are equally good mark algorithm 1. ")
    P=zeros(4,1);
    R=zeros(4,1);
    F=zeros(4,1);
    A=['A','B','C','D'];
    for i=1:4
        fprintf("\n\nAlgorithm %d\n",i)
        a=randi([50,100]);
        B=randi([10,50]);
        c=randi([10,50]);
        d=randi([50,100]);
        fprintf("\t\t\t             Actual class :1  Actual class :0\n")
        fprintf("Predicted class :1    %d             %d\n",a,B)
        fprintf("Predicted class :0    %d             %d",c,d)
        
        %Code to calculate Precision Value.
        P(i)=a/(a+B);
        
        %Code to calculate Recall Value.
        R(i)=a/(a+c);
        
        %Code to calculate Fscore.
        F(i)=2*P(i)*R(i)/(P(i)+R(i));
    end
    fprintf("\nA. Algorithm 1\nB. Algorithm 2\nC. Algoritm 3\nD. Algorithm 4\nE. None of theses")
    [M,I] = max(F,[],'linear');
    fprintf("\nANSWER = %s",A(I));
    fprintf("\nSOLUTION\n")
    fprintf("We need to calculate F Score in each cases and select the option with maximum F score\n")
    fprintf("F score = 2*Precision* Recall/(Precison + Recall)\n")
    fprintf("Precison=True Positive/(True Positive+False Postive)\n)")
    P
    fprintf("Recall=True Positive/(True positive+False Negative)\n")
    R
    fprintf("F score = 2*Precision* Recall/(Precison + Recall)\n")
    F
    fprintf("Maximum F score is in algorithm %d, which is our answer\n\n",I)
end



%Question 2
for i=1:5
    theta=rand(3,3);
    x1=rand();
    x2=rand();
    r=randi([1,3]);
    A=['A','B','C'];
    fprintf("\nQ 2 V %d\n",i)
    fprintf("In the following neural network representation (All nodes are inter connected). Calculate the value in node %d of Layer 2. The neural network uses sigmoid function for activation\n", r)
    fprintf("1          a21\n")
    fprintf("x1 ------> a22 ---------> a31\n")
    fprintf("x2         a23\n")
    fprintf("The parameter values are given such that Layer2(3x1)=theta(3x3)*(Layer1)(3x1)\n")
    theta
    x1,x2
    layer1= [1; x1 ;x2];
    layer2_=theta*layer1;
    layer2=sigmoid(layer2_);
    fprintf("A.%d\nB.%d\nC.%d\nD.%d\nE.None of these\n",layer2(1),layer2(2),layer2(3),r)
    fprintf("SOLUTION\n")
    fprintf("ANSWER=%s\n",A(r))
    fprintf("The values in layer 2 before activation is given by theta*Layer1\n")
    layer2_
    fprintf("After applying activation g(z)=1/(1+e^-z)\n")
    layer2
    
end

%Question 3
for i=1:5
    fprintf("Q 3 V %d\n",i);
    %A set of 10 statements out of which 5 are true.
    QMatrix={'In a machine learning algorithm, if the number of parameters grow with the amount of training data, then the model is non-parametric.','Decision trees can be used for clustering.','Principal component analysis is an example of a deterministic algorithm.','If we know that the conditional independence assumptions made by Naïve Bayes are not true for our problem, and we have lots of training data, we might prefer Logistic Regression over Naive Bayes for a particular learning task.','Movie recommendation systems are an example of clustering and classification.','If you train linear regression estimator with only half the data, the bias will be smaller.','Reinforcement learning is an example of supervised learning.','The claim by a project team that their method is good based on the low training error that they reported is correct.','Increase in the number of training examples in logistic regression will eventually decrease the Bias and increase the Variance.','It is necessary to have a target variable for applying dimensionality reduction algorithms.'};
    %A set of 4 statements for options out of which some are true.
    OMatrix=QMatrix(randperm(10, 4));
    %question
    fprintf('Which of the following options are correct?\n');
    fprintf('A)');
    disp(OMatrix(1,1))
    fprintf('B)');
    disp(OMatrix(1,2));
    fprintf('C)');
    disp(OMatrix(1,3));
    fprintf('D)');
    disp(OMatrix(1,4));
    fprintf('E)\tNone of the above\n')
    fprintf("\nAnswer:The option(s) ");
    c=0;
    %checking which options are correct and then displaying it.
    for j=1:4
        for k=1:5
            l=strcmp(OMatrix(1,j),QMatrix(1,k));
            c=c+l;
            if l==1
                if j==1
                    fprintf('A)')
                end
                if j==2
                    fprintf('B)')
                end
                if j==3
                    fprintf('C)')
                end
                if j==4
                    fprintf('D)')
                end
            end

        end
    end
    if c==0
        fprintf('E)');
    end
    fprintf(' is/are correct.\n\n\n');
end
%Question 4
for i = 1:5
    fprintf('Q4 ,V %d \n',i);
    global phi
    global m
    global b
    
    %generate a randon real number between 3 and 4 for testing
    test_time = 3 + rand(1,1);
    fprintf(['The following is a data set on the hours spend by each student on the MFDS course project ' ...
        'and the percentage of mark they got in percentage for the project work.Use a Linear regression –ordinary ' ...
        'least square technique to estimate the parameters and then use a gradient descent algorithm to improve the estimation.\n' ...
        'Consider the best fit line as y= m*x + b and improve the prediction by using the following approach and iterate though ' ...
        'all sample data provided.\n m = m + (error*x) *learning rate \n b = b + (error) * learning rate\nGiven Learning rate = 0.1 and (error = y –prediction in that iteration) \n'])
    hours_spend = [1;2.5;2.8;3.2;4.3;5];
    percentage_mark = [27.8;41.7;44.4;50.1;60.8;66.6];
    T=table(hours_spend,percentage_mark);
    disp(T)
    fprintf('Predict the percentage mark scored by a student who spend %f hours\n\n',test_time);
    
   
    x = hours_spend;
    y = percentage_mark;
    
    %calling the function to calculate the prediction
    Prediction_test = Predictor(x,y,test_time);
    
    option1 = num2str(Prediction_test+rand(1,1));
    option2 = num2str(Prediction_test);
    option3 = num2str(Prediction_test+rand(1,1));
    option4 = num2str(Prediction_test+rand(1,1)); 
    
    fprintf('A) %s \n',option1);
    fprintf('B) %s \n',option2);
    fprintf('C) %s \n',option3);
    fprintf('D) %s \n',option4);
    
    fprintf('the correct option is B) %f \n\n',Prediction_test);
    
    % print out parameters after least square method
    
    fprintf('The parameters[b,m] after least square method are \n');
    phi
    % print out new values of parameters after gradient descent algorithm
    fprintf('new b value after gradient descent is \n');
    disp(b)
    fprintf('new m value after gradient descent is \n');
    disp(m)
    
    fprintf('the predicted percentage mark for %f hours spend is %f\n\n',test_time,Prediction_test);
end

%function to make the prediction

function prediction = Predictor(x,y,test)
    global phi
    global m
    global b
    %method to calculate parameters by least square method
    N = length(x);
    A = [N, sum(x); sum(x),sum(x.*x)];
    B = [sum(y); sum(x.*y)];
    phi = inv(A)*B;
    
    b = phi(1);
    m = phi(2);
    learningRate = 0.1;
    %running loop to iterate over the samples using the equation given
    for i=1:N
        X = x(i);
        Y = y(i);
        guess = m*X + b ;
        error = Y - guess;
        
        m= m +(error*X) * learningRate;
        b= b +(error) * learningRate;
    end
    %calculating the prediction value by new m and b
    prediction = m * test + b;
    
end
%Function to calculate sigmoid for question 2
function g = sigmoid(z)
        g = zeros(size(z));
        g = 1.0 ./ ( 1.0 + exp(-z));
end