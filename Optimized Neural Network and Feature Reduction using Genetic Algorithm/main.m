%% 遗传算法的优化计算——输入自变量降维

%% 清空环境变量
clear all
clc
warning off
%% 声明全局变量
global P_train T_train P_test T_test  mint maxt S s1
S = 6;%自变量数量
s1 = 50;
%% 导入数据
load PlateData01.mat %Assume SAR value larger than 86 is high
a = randperm(576);
Train = PlateData(a(1:500),:);
Test = PlateData(a(501:end),:);
% 训练数据
P_train = Train(:,3:end)';
T_train = Train(:,2)';
% 测试数据
P_test = Test(:,3:end)';
T_test = Test(:,2)';
% 显示实验条件
total_B = length(find(PlateData(:,2) == 1));
total_M = length(find(PlateData(:,2) == 2));
count_B = length(find(T_train == 1));
count_M = length(find(T_train == 2));
number_B = length(find(T_test == 1));
number_M = length(find(T_test == 2));
disp('Epxeriment Environment：');
disp(['Total Data：' num2str(569)...
      '  High Peak SAR：' num2str(total_B)...
      '  Low Peak SAR：' num2str(total_M)]);
disp(['Training：' num2str(500)...
      '  High：' num2str(count_B)...
      '  Low：' num2str(count_M)]);
disp(['Testing：' num2str(69)...
      '  High：' num2str(number_B)...
      '  Low：' num2str(number_M)]);
%% 数据归一化
[P_train,minp,maxp,T_train,mint,maxt] = premnmx(P_train,T_train);
P_test = tramnmx(P_test,minp,maxp);
%% 创建单BP网络
t = cputime;
net_bp = newff(minmax(P_train),[s1,1],{'tansig','purelin'},'trainlm');
% 设置训练参数
net_bp.trainParam.epochs = 1000;
net_bp.trainParam.show = 10;
net_bp.trainParam.goal = 0.1;
net_bp.trainParam.lr = 0.1;
net_bp.trainParam.showwindow = 0;
%% 训练单BP网络
net_bp = train(net_bp,P_train,T_train);
%% 仿真测试单BP网络
tn_bp_sim = sim(net_bp,P_test);
% 反归一化
T_bp_sim = postmnmx(tn_bp_sim,mint,maxt);
e = cputime - t;
T_bp_sim(T_bp_sim > 1.5) = 2;
T_bp_sim(T_bp_sim < 1.5) = 1;
result_bp = [T_bp_sim' T_test'];
%% 结果显示（单BP网络）
number_B_sim = length(find(T_bp_sim == 1 & T_test == 1));
number_M_sim = length(find(T_bp_sim == 2 &T_test == 2));
disp('(1)BP Network Result：');
disp(['High Peak SAR：' num2str(number_B_sim)...
      '  Mistaken：' num2str(number_B - number_B_sim)...
      '  Positive Rate p1 = ' num2str(number_B_sim/number_B*100) '%']);
disp(['Low Peak SAR：' num2str(number_M_sim)...
      '  Mistaken：' num2str(number_M - number_M_sim)...
      '  Positive Rate p2 = ' num2str(number_M_sim/number_M*100) '%']);
disp(['Time of module running：' num2str(e) 's'] );
%% 遗传算法优化
popu = 20;  
bounds = ones(S,1)*[0,1];
% 产生初始种群
initPop = randi([0 1],popu,S);
% 计算初始种群适应度
initFit = zeros(popu,1);
for i = 1:size(initPop,1)
    initFit(i) = de_code(initPop(i,:));
end
initPop = [initPop initFit];
gen = 100; 
% 优化计算
[X,EndPop,BPop,Trace] = ga(bounds,'fitness',[],initPop,[1e-6 1 0],'maxGenTerm',...
    gen,'normGeomSelect',0.09,'simpleXover',2,'boundaryMutation',[2 gen 3]);
[m,n] = find(X == 1);
disp(['Features selected after optimization:' num2str(n)]);
% 绘制适应度函数进化曲线
figure
plot(Trace(:,1),Trace(:,3),'r:')
hold on
plot(Trace(:,1),Trace(:,2),'b')
xlabel('Number of Evolution')
ylabel('Fitness Function')
title('Plot of Fitness Function Evolution ')
legend('Everage Fitness','Best Fitness')
xlim([1 gen])
%% 新训练集/测试集数据提取
p_train = zeros(size(n,2),size(T_train,2));
p_test = zeros(size(n,2),size(T_test,2));
for i = 1:length(n)
    p_train(i,:) = P_train(n(i),:);
    p_test(i,:) = P_test(n(i),:);
end
t_train = T_train;
%% 创建优化BP网络
t = cputime;
net_ga = newff(minmax(p_train),[s1,1],{'tansig','purelin'},'trainlm');
% 训练参数设置
net_ga.trainParam.epochs = 1000;
net_ga.trainParam.show = 10;
net_ga.trainParam.goal = 0.1;
net_ga.trainParam.lr = 0.1;
net_ga.trainParam.showwindow = 0;
%% 训练优化BP网络
net_ga = train(net_ga,p_train,t_train);
%% 仿真测试优化BP网络
tn_ga_sim = sim(net_ga,p_test);
% 反归一化
T_ga_sim = postmnmx(tn_ga_sim,mint,maxt);
e = cputime - t;
T_ga_sim(T_ga_sim > 1.5) = 2;
T_ga_sim(T_ga_sim < 1.5) = 1;
result_ga = [T_ga_sim' T_test'];
%% 结果显示（优化BP网络）
number_b_sim = length(find(T_ga_sim == 1 & T_test == 1));
number_m_sim = length(find(T_ga_sim == 2 &T_test == 2));
disp('(2)Optimized Network Result：');
disp(['Number of High Peak SAR：' num2str(number_b_sim)...
      '  Mistaken：' num2str(number_B - number_b_sim)...
      '  Positive Rate p1=' num2str(number_b_sim/number_B*100) '%']);
disp(['Number of Low Peak SAR：' num2str(number_m_sim)...
      '  Mistaken：' num2str(number_M - number_m_sim)...
      '  Positive Rate p2=' num2str(number_m_sim/number_M*100) '%']);
disp(['Time of running：' num2str(e) 's'] );
