function [L, DrawL, R, DrawR] = calDraw(data1, data2, off1, off2)
if data1(1) < data2(1)
    L = data1;
    R = data2;
    DrawL = data1+off1;
    DrawR = data2+off2;
else
    L = data2;
    R = data1;
    DrawL = data2+off1;
    DrawR = data1+off2;
end