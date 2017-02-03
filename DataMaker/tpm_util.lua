local M = {}


function M.helloWorld()
	print("Hello World!!!!")
end 

function M.padRightDownCorner(image)
	-- body
	local h = image:size(2)
	local w = image:size(3)
print(h)
print(w)



	local padding = {}
	padding[1] = 0 --up
	padding[2] = 0 --left

	if h%8 == 0 then padding[3] = 0 --down
	else padding[3] = 8 - (h%8)
	end
	if w%8 == 0 then padding[4] = 0 --right
	else padding[4] = (8 - (w%8))
	end

	local pix  = 0.5
	local ndim = image:dim()

	local s = nn.Sequential()
	  :add(nn.Padding(ndim-1,  padding[1], ndim, pix))
	  :add(nn.Padding(ndim-1,  padding[2], ndim, pix))
	  :add(nn.Padding(ndim,    padding[3], ndim, pix))
	  :add(nn.Padding(ndim,    padding[4], ndim, pix))

	local y = s:forward(image)
	return y, padding

end

function M.copyLayer(layer_id, torch_model, layer_name, caffe_model)
	-- body
	for  _ , m  in pairs(caffe_model:listModules()) do
		if m.name == layer_name then
			torch_model.modules[layer_id].weight = m.weight:clone()
			torch_model.modules[layer_id].bias = m.bias:clone()
			-- print(torch_model.modules[layer_id])
			-- print(m)
			print("successful copy " .. layer_name ..'   '.. torch.type(torch_model.modules[layer_id]))
			return
		end
	end
	print('ERROR Cannnot Copy Layer'.. layer_name)
end


function M.localMaximumDetection(img, half_window_size, threshold)
	res = {}
	for i = 1 + half_window_size, person_map_resized:size()[2] - half_window_size do
    	for j = 1 + half_window_size, person_map_resized:size()[3] - half_window_size do
        	local region = person_map_resized[{{1},{i - half_window_size, i + half_window_size},{j - half_window_size, j + half_window_size}}]   
        	if person_map_resized[1][i][j] > threshold and person_map_resized[1][i][j] == torch.max(region) then
            	print('local maximum: ['.. i .. ',  '..j..']')
            	res[table.getn(res)+1] = {i, j}
        	end    
    	end
	end
	return res
end

-- function getJetColor(v,vmin,vmax)
-- 	-- body
-- 	c = {0,0,0}
-- 	if torch.lt(v,vmin) then
-- 		v = vmin
-- 	end
-- 	if torch.gt(v,vmax) then
-- 		v = vmax
-- 	end
-- 	dv = vmax - vmin
-- 	if torch.lt(v, (vmin + 0.125 * dv)) then
-- 		c[1] = 256 * (0.5 + (v * 4))  --B: 0.5 ~ 1
-- 	elseif torch.lt(v, (vmin + 0.375 * dv)) then
-- 		c[1] = 255					
-- 		c[2] = 256 * (v - 0.125) * 4 --G: 0 ~ 1
-- 	elseif torch.lt(v < (vmin + 0.625 * dv)) then
-- 		c[1] = 256 * (-4 * v + 2.5)
-- 		c[2] = 255
-- 		c[3] = 256 * (4 * (v - 0.375)) 
-- 	elseif torch.lt(v, (vmin + 0.875 * dv)) then
-- 		c[2] = 256 * (-4 * v + 3.5)
-- 		c[3] = 255
-- 	else
-- 		c[3] =  256 * (-4 * v + 4.5)
-- 	end

-- 	return c
-- end

function getJetColor(v,vmin,vmax)
	-- body
	c = torch.Tensor(3,1):zero()
	if v < vmin then
		v = vmin
	end
	if v > vmax then
		v = vmax
	end
	dv = vmax - vmin
	if v < (vmin + 0.125 * dv) then
		c[1] = 256 * (0.5 + (v * 4))  --B: 0.5 ~ 1
	elseif v < (vmin + 0.375 * dv) then
		c[1] = 255					
		c[2] = 256 * (v - 0.125) * 4 --G: 0 ~ 1
	elseif v <  (vmin + 0.625 * dv) then
		c[1] = 256 * (-4 * v + 2.5)
		c[2] = 255
		c[3] = 256 * (4 * (v - 0.375)) 
	elseif v < (vmin + 0.875 * dv) then
		c[2] = 256 * (-4 * v + 3.5)
		c[3] = 255
	else
		c[3] =  256 * (-4 * v + 4.5)
	end

	return c

end


function M.colorize(gray_img)
	-- body
	out = torch.Tensor(3, gray_img:size()[2], gray_img:size()[3]):zero()
	for i = 1, gray_img:size()[2] do
		for j = 1, gray_img:size()[3] do
			out[{{},{i},{j}}] = getJetColor(gray_img[1][i][j],0,1)
		end
	end
	return out
end

return M

