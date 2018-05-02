% This function is Bayesian classifier used for classification of two classes of patterns
w1 = [-3.9847 -3.5549 -1.2401 -0.9780 -0.7932 -2.8531 -2.7605 -3.7287 -3.5414 -2.2692 -3.4549 -3.0752 -3.9934 -0.9780 -1.5799 -1.4885 -0.7431 -0.4221 -1.1186 -2.3462 -1.0826 -3.4196 -1.3193 -0.8367 -0.6579 -2.9683];
w2 = [2.8792 0.7932 1.1882 3.0682 4.2532 0.3271 0.9846 2.7648 2.6588];
% calculate prior  probability density functions
aver1=mean(w1);
aver2=mean(w2);
s1=std(w1);
s2=std(w2);
x=-5:0.1:5;
f1=(1 / (sqrt(2 * pi) * s1)) * exp(-1 * (x - aver1) .^ 2 / (2 * s1 ^ 2));
f2=(1 / (sqrt(2 * pi) * s2)) * exp(-1 * (x - aver2) .^ 2 / (2 * s2 ^ 2));
plot(x,f1,'r');
hold on
plot(x,f2,'g');
title('Prior Probability Density Functions');
% calculate posterior probability density functions
post_f1=(f1*0.9)./(f1*0.9+f2*0.1);
post_f2=(f2*0.1)./(f1*0.9+f2*0.1);
figure;
plot(x,post_f1,'r');
hold on
plot(x,post_f2,'g');
title('Posterior Probability Density Functions');
for i=1:length(x)
    if(post_f1(i)<0.5)
        disp(['Decision Boundary with maximum accuracy: ', num2str(x(i))])
        break
    end
end
% Consider the decision loss
r21=6;
r12=1;
l1=post_f2*r12;
l2=post_f1*r21;
figure;
hold on
plot(x,l1,'r');
plot(x,l2,'g');
title('Desicion Loss');
for i=1:length(x)
    if(l1(i)>l2(i))
        disp(['Decision Boundary with minimal risk: ', num2str(x(i))]);
        break
    end
end
