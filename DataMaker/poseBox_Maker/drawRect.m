function drawRect(pt)
line([pt{1}(1), pt{2}(1)], [pt{1}(2), pt{2}(2)], 'Color', 'g');
line([pt{3}(1), pt{4}(1)], [pt{3}(2), pt{4}(2)], 'Color', 'g');
line([pt{1}(1), pt{3}(1)], [pt{1}(2), pt{3}(2)], 'Color', 'g');
line([pt{2}(1), pt{4}(1)], [pt{2}(2), pt{4}(2)], 'Color', 'g');