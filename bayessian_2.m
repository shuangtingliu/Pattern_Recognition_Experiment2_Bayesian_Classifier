% This function is Bayesian classifier used for classification of mutiple classes of patterns
% Parameters
prob1 = 0.3;
prob2 = 0.1;
prob3 = 0.6;
risk = [0 2 6; 3 0 7; 2 1 0];
%set the average value of data
u1 = [0 0]';
u2 = [5 5]';
u3 = [1 3]';
%covariance matrix
Sigma1 = eye(2)*0.7;
Sigma2 = eye(2)*0.3;
Sigma3 = eye(2)*0.5;
%generate points
p1 = mvnrnd(u1,Sigma1,100);
p2 = mvnrnd(u2,Sigma2,100);
p3 = mvnrnd(u3,Sigma3,100);
figure;
plot(p1(:,1),p1(:,2),'b.',u1(1),u1(2),'r*','MarkerSize',10);
hold on
plot(p2(:,1),p2(:,2),'g.',u2(1),u2(2),'r*','MarkerSize',10);
plot(p3(:,1),p3(:,2),'k.',u3(1),u3(2),'r*','MarkerSize',10);
title('Dataset');
% calculate prior  probability density functions
aver1 = mean(p1);
aver2 = mean(p2);
aver3 = mean(p3);
cov1 = cov(p1(:, 1), p1(:, 2));
cov2 = cov(p2(:, 1), p2(:, 2));
cov3 = cov(p3(:, 1), p3(:, 2));
[x, y] = meshgrid(-5 : 0.1 : 10);
f1=reshape(mvnpdf([x(:), y(:)], aver1, cov1), size(x));
f2=reshape(mvnpdf([x(:), y(:)], aver2, cov2), size(x));
f3=reshape(mvnpdf([x(:), y(:)], aver3, cov3), size(x));
figure;
surf(x, y, f1);
hold on
surf(x, y, f2);
surf(x, y, f3);
title('Prior Probability Density Functions');
% calculate posterior probability density functions
post_f1 = (f1 * prob1)./(f1 * prob1 + f2 * prob2 + f3 * prob3);
post_f2 = (f2 * prob2)./(f1 * prob1 + f2 * prob2 + f3 * prob3);
post_f3 = (f3 * prob3)./(f1 * prob1 + f2 * prob2 + f3 * prob3);
figure;
surf(x, y, post_f1);
hold on
surf(x, y, post_f2);
surf(x, y, post_f3);
title('Posterior Probability Density Functions');
% Decision Boundary with maximum accuracy:
Max=zeros(length(x),length(y));
for i=1:length(x)
    for j=1:length(y)
        if(post_f1(i,j)>post_f2(i,j))
            if(post_f1(i,j)>post_f3(i,j))
                Max(i,j)=0;
            elseif(post_f3(i,j)>=post_f1(i,j))
                Max(i,j)=2;
            end
        else
            if(post_f2(i,j)>post_f3(i,j))
                Max(i,j)=1;
            elseif(post_f3(i,j)>=post_f2(i,j))
                Max(i,j)=2;
            end
        end
    end
end
figure;
hold on
surf(x, y, Max);
title('Decision Boundary with maximum accuracy');

% Consider the decision loss
l1=post_f2*risk(1,2)+post_f3*risk(1,3);
l2=post_f1*risk(2,1)+post_f3*risk(2,3);
l3=post_f1*risk(3,1)+post_f2*risk(3,2);
figure;
surf(x, y, l1);
hold on
surf(x, y, l2);
surf(x, y, l3);
title('Desicion Loss');


% Decision Boundary with with minimal risk:
Min=zeros(length(x),length(y));
for i=1:length(x)
    for j=1:length(y)
        if(l1(i,j)<l2(i,j))
            if(l1(i,j)<l3(i,j))
                Min(i,j)=0;
            elseif(l3(i,j)<=l1(i,j))
                Min(i,j)=2;
            end
        else
            if(l2(i,j)<l3(i,j))
                Min(i,j)=1;
            elseif(l3(i,j)<=l2(i,j))
                Min(i,j)=2;
            end
        end
    end
end
figure;
hold on
surf(x, y, Min);
title('Decision Boundary with minimal risk');
