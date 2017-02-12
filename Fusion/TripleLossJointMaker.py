import caffe
import numpy as np

image_res50 = caffe.Net('res50_img_baseline_deploy.prototxt', '../Models/ResNet_OrigImg__iter_40000.caffemodel', caffe.TEST)
pose_res50 = caffe.Net('res50_pose_baseline_deploy.prototxt', '../Models/ResNet_PoseBox__iter_40000.caffemodel', caffe.TEST)

tripleLoss_resnet50 = caffe.Net('./res50_tripleLoss.prototxt','../Models/ResNet_PoseBox__iter_40000.caffemodel',caffe.TEST)

layer_name_list = image_res50.params.keys()

for l in layer_name_list:
	for i in range(0,len(image_res50.params[l])):
		tripleLoss_resnet50.params['origImg_'+l][i].data[...] = image_res50.params[l][i].data[...]

layer_name_list = pose_res50.params.keys()

for l in layer_name_list:
	for i in range(0,len(pose_res50.params[l])):
		tripleLoss_resnet50.params['poseBox_'+l][i].data[...] = pose_res50.params[l][i].data[...]

tripleLoss_resnet50.save('res50-TripleLoss2-init.caffemodel')
