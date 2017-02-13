import os

### generate new prototxt
with open('res50_train_val_dy.prototxt') as fin:
	with open( 'resnet_pose.prototxt', 'wb') as fout:
		for line in fin:
			line = line.replace('{TRAIN_PATH}', './DataMaker/hdf5_train_data_list.txt')
			line = line.replace('{TEST_PATH}', './DataMaker/hdf5_test_data_list.txt')
			line = line.replace('{LABEL}', 'label')
			line = line.replace('{DATA}', 'pose_data')
			fout.write(line)


with open('res50_train_val_dy.prototxt') as fin:
        with open( 'resnet_orig.prototxt', 'wb') as fout:
                for line in fin:
                        line = line.replace('{TRAIN_PATH}', './DataMaker/hdf5_train_data_list.txt')
                        line = line.replace('{TEST_PATH}', './DataMaker/hdf5_test_data_list.txt')
                        line = line.replace('{LABEL}', 'label')
                        line = line.replace('{DATA}', 'orig_data')
                        fout.write(line)

                        
### generate new solver
with open('res50_solver_dy.prototxt') as fin:
	with open( 'res50_solver_pose.prototxt', 'wb') as fout:
		for line in fin:
			line = line.replace('{SNAPSHOT}', './trainlog/ResNet_pose16_')
                        line = line.replace('{NET}','./resnet_pose.prototxt')
                        fout.write(line)

with open('res50_solver_dy.prototxt') as fin:
        with open( 'res50_solver_orig.prototxt', 'wb') as fout:
                for line in fin:
                        line = line.replace('{SNAPSHOT}', './trainlog/ResNet_orig16_')
                        line = line.replace('{NET}','./resnet_orig.prototxt')
                        fout.write(line)

                        
#mk new snapshot dir
os.mkdir('./trainlog')

### run training
os.system('~/library/caffe/build/tools/caffe train --solver ./res50_solver_pose.prototxt --weights ./Models/ResNet-50-model.caffemodel --gpu=1 2>&1 | tee ./trainlogs/ResNet_pose16_' )

os.system('~/library/caffe/build/tools/caffe train --solver ./res50_solver_orig.prototxt --weights ./Models/ResNet-50-model.caffemodel --gpu=1 2>&1 | tee ./trainlogs/ResNet_orig16_' )
