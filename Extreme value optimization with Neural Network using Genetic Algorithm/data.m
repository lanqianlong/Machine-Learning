for i=1:576
    input(i,:)=10*rand(1,6)-5;
    output(i)=input(i,1)^2+input(i,6)^2;
end
output=output';

save data1 input output