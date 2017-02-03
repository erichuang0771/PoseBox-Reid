import os

### generate new prototxt
with open('res50_train_val_dy.prototxt') as fin:
	with open( 'resnet_pose.prototxt', 'wb') as fout:
		for line in fin:
			line = line.replace('{TRAIN_PATH}', '/home/eric/re-id/train/Caffe/eric_resNet50_Joint/eric_dataset/train_hdf5.txt')
			line = line.replace('{TEST_PATH}', '/home/eric/re-id/train/Caffe/eric_resNet50_Joint/eric_dataset/val_hdf5.txt')
			line = line.replace('{LABEL}', 'label')
			line = line.replace('{DATA}', 'pose_data')
			fout.write(line)

### generate new solver
with open('res50_solver_dy.prototxt') as fin:
	with open( 'res50_solver_pose.prototxt', 'wb') as fout:
		for line in fin:
			line = line.replace('{SNAPSHOT}', '/media/eric/b13d4a46-a007-4c3c-b23e-c198cf4899c0/home/mscv/Data/new_high/snapshot_poseBox_liangBaseline16/ResNet_pose16_')
			fout.write(line)

#mk new snapshot dir
os.mkdir('/media/eric/b13d4a46-a007-4c3c-b23e-c198cf4899c0/home/mscv/Data/new_high/snapshot_poseBox_liangBaseline16')

### run training
os.system('~/caffe/build/tools/caffe train --solver ./res50_solver_now2.prototxt --weights ./ResNet-50-model.caffemodel --gpu=1 2>&1 | tee ./logs/ResNet_pose16_' )
:x

