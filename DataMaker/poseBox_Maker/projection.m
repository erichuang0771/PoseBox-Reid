function imgn = projection(p1, p2, img, w, h)

img2 = zeros(h, w);
[h, w] = size(img2);

tform = maketform('projective',p1,p2);
T2 = calc_homography(p1,p2);   %计算单应性矩阵
T = maketform('projective',T2);   %投影矩阵

imgn = imtransform(img, T,'size', size(img2), 'XData', [1, w], 'YData', [1, h]);     %投影
% figure;imshow(imgn);