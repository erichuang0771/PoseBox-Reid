function point = calheadneck(neck,head, len)
if head(2) == neck(2)
%     head(2) = head(2)-1;
    neck(2) = neck(2)+1;
end
% k = (knee(2) - hip(2)+.0000001)/(knee(1) - hip(1)+.0000001);
k2 = 0;
xoffset = len/2;
% yoffset = len*(abs(k2)/sqrt(1+k2^2));
if neck(1) < head(1)
    point{1} = [max(0, neck(1)-xoffset), max(0, head(2))];
    point{2} = [max(0, head(1)+xoffset), max(0, head(2))];
    point{3} = [max(0, neck(1)-xoffset), max(0, neck(2))];
    point{4} = [max(0, head(1)+xoffset), max(0, neck(2))];
else
    point{1} = [max(0, head(1)-xoffset), max(0, head(2))];
    point{2} = [max(0, neck(1)+xoffset), max(0, head(2))];
    point{3} = [max(0, head(1)-xoffset), max(0, neck(2))];
    point{4} = [max(0, neck(1)+xoffset), max(0, neck(2))];
end




