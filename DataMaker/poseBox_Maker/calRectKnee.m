function point = calRectKnee(knee, hip, len)
if knee(1) == hip(1)
    knee(1) = knee(1)-1;
    hip(1) = hip(1)+1;
end
k = (knee(2) - hip(2)+.0000001)/(knee(1) - hip(1)+.0000001);
k2 = -1/k;
xoffset = len*(1/sqrt(1+k2^2));
yoffset = len*(abs(k2)/sqrt(1+k2^2));
point{1} = [max(0, hip(1)-xoffset), max(0, hip(2)-yoffset*sign(k2))];
point{2} = [max(0, hip(1)+xoffset), max(0, hip(2)+yoffset*sign(k2))];
point{3} = [max(0, knee(1)-xoffset), max(0, knee(2)-yoffset*sign(k2))];
point{4} = [max(0, knee(1)+xoffset), max(0, knee(2)+yoffset*sign(k2))];




