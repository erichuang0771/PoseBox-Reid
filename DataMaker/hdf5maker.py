import cv2
import numpy as np
import lmdb
import sys
import glob
import errno
import os
import csv
import random
import h5py
import caffe
import argparse 

class ReIDDataLayer():
	def __init__(self, orginal_dir, posebox_dir, feature_dir, testOrTrain):
		orginal_dir = orginal_dir
		posebox_dir = posebox_dir
		feature_dir = feature_dir
		self.testOrTrain = testOrTrain
		
		file_list = glob.glob(posebox_dir + '/*.jpg')
	 	orginal_file_list = glob.glob(orginal_dir + '/*.jpg')
	 	assert(len(file_list) == len(orginal_file_list))

		# initalize fileID 2 personID mapping
		fileID2labelID = {}
		filedIDCounter = {}
		with open(feature_dir) as csvfile:
	 		reader = csv.DictReader(csvfile)
		 	for row in reader:
		 		file_name = row['file_name']
		 		if filedIDCounter.get(file_name[0:4]) == None:
					filedIDCounter[file_name[0:4]] = 1.0
				else:
					filedIDCounter[file_name[0:4]] = filedIDCounter[file_name[0:4]] + 1.0

		for key in filedIDCounter:
			#print(filedIDCounter[key])
			filedIDCounter[key] = filedIDCounter[key] * 0.9



		# build datalist
		file_counter = 0
		self.train_data_list = list()
		self.test_data_list = list()
		all_data_list = list()

	 	counter = 0
		with open(feature_dir) as csvfile:
	 		reader = csv.DictReader(csvfile)
		 	for row in reader:
		 		file_name = row['file_name']

		 		data_entry = list()
		 		data_entry.append( posebox_dir+'/'+file_name )

		 		data_entry.append( orginal_dir+'/'+file_name )

		 		conf = np.zeros((12,))

		 		#symmtric configureation
				conf[0] = row['Rshoulder_confidence']
				conf[1] = row['Rhip_confidence']
				conf[2] = row['Rknee_confidence']
				conf[3] = row['Rankle_confidence']
				conf[4] = row['Relbow_confidence']
				conf[5] = row['Rwrist_confidence']

				conf[6] = row['Lwrist_confidence']
				conf[7] = row['Lelbow_confidence']
				conf[8] = row['Lankle_confidence']
				conf[9] = row['Lknee_confidence']
				conf[10] = row['Lhip_confidence']
				conf[11] = row['Lshoulder_confidence']
				data_entry.append(conf.reshape((1,12)))

		 		if fileID2labelID.get(file_name[0:4]) == None:
					fileID2labelID[file_name[0:4]] = len(fileID2labelID)+1

		 		# label = np.zeros((751,))
		 		# label[fileID2labelID[file_name[0:4]] -1 ] = 1
		 		label = np.zeros((1,)).astype(int)
		 		label[0] = fileID2labelID[file_name[0:4]] -1
		 		data_entry.append(label.reshape(1,1))
		 		
		 		data_entry.append(file_name)

		 		all_data_list.append(data_entry)

		 		counter = counter + 1

		# random val & train data

		# random.shuffle(all_data_list)
		for entry in all_data_list:
		 	file_name = entry[-1]
		 	if filedIDCounter[file_name[0:4]] > 0:
		 		self.train_data_list.append(entry)
		 		filedIDCounter[file_name[0:4]] = filedIDCounter[file_name[0:4]] - 1
		 	else:
		 		self.test_data_list.append(entry)

		print('Test Data List Size: '+ str(len(self.test_data_list)))
		print('Train Data List Size: '+ str(len(self.train_data_list)))
		

	def size(self):
		if self.testOrTrain == 'train':
			return len(self.train_data_list)
		else:
			return len(self.test_data_list)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
        parser.add_argument('--orig_data', help='the path of original data')
        parser.add_argument('--pose_data', help='the path of posebox data')
        parser.add_argument('--csv_list', help='the csv file that stores the confidence scores of posebox')
        parser.add_argument('--output_dir', help='the output path of all hdf5 data')
        args = parser.parse_args()
        assert args.orig_data and args.pose_data and args.csv_list and args.output_dir
        print args.orig_data
        print args.pose_data
        print args.csv_list
        print args.output_dir

        
        
        # reid_home_dir = '/home/eric/re-id'
	a = ReIDDataLayer(args.orig_data,
			  args.pose_data,
			  args.csv_list,
			  'train')
	counter = 0
	print(len(a.train_data_list))
	print(len(a.test_data_list))
	
	counter = 0

	txt_file = open("./hdf5_train_data_list.txt", "w")


	blob = caffe.proto.caffe_pb2.BlobProto()
	mean_data = open( './ResNet_mean.binaryproto' , 'rb' ).read()
	blob.ParseFromString(mean_data)
	arr = np.array( caffe.io.blobproto_to_array(blob) )[0]

	data_transformer = caffe.io.Transformer({'data': (1,3,224,224)})
	data_transformer.set_transpose('data', (2,0,1))
	data_transformer.set_mean('data', arr.mean(1).mean(1)) # mean pixel
	data_transformer.set_raw_scale('data', 255) # the reference model operates on images in [0,255] range instead of [0,1]
	data_transformer.set_channel_swap('data', (2,1,0)) # the reference model has channels in BGR order instead of RGB


	for k in a.train_data_list:
		pose = data_transformer.preprocess('data', caffe.io.load_image(k[0])).reshape((1,3,224,224))
		orig = data_transformer.preprocess('data', caffe.io.load_image(k[1])).reshape((1,3,224,224))
		
		conf = k[2]
		label = k[3]
		name = k[4]
		# print(label.shape)
		path_ = args.output_dir + '/' + name[0:-4]+'.h5'
		#print path_
		hf = h5py.File(path_, 'w')
		hf.create_dataset('/pose_data', data=pose)
		hf.create_dataset('/orig_data', data=orig)
		hf.create_dataset('/conf', data=conf)
		hf.create_dataset('/label', data=label)
		a_path = os.path.abspath(path_)
		print(a_path)
		txt_file.write(args.output_dir + '/' + name[0:-4]+'.h5\n')



		########### MIRROR
		for c in range(0,3):
			pose[0,c,:,:] = np.fliplr(pose[0,c,:,:])
			orig[0,c,:,:] = np.fliplr(orig[0,c,:,:])
		conf = np.fliplr(conf)
		path_ = args.output_dir + '/' + name[0:-4]+'_m.h5'
		hf = h5py.File(path_, 'w')
		hf.create_dataset('/pose_data', data=pose)
		hf.create_dataset('/orig_data', data=orig)
		hf.create_dataset('/conf', data=conf)
		hf.create_dataset('/label', data=label)
		a_path = os.path.abspath(path_)
		print(a_path)
		txt_file.write(args.output_dir + '/' + name[0:-4]+'_m.h5\n')
		
		counter = counter + 1	

	print counter
	print len(a.train_data_list)


        ####### handle val data
        counter = 0
	txt_file = open("./hdf5_test_data_list.txt", "w")

	for k in a.test_data_list:
		pose = data_transformer.preprocess('data', caffe.io.load_image(k[0])).reshape((1,3,224,224))
		orig = data_transformer.preprocess('data', caffe.io.load_image(k[1])).reshape((1,3,224,224))
		
		conf = k[2]
		label = k[3]
		name = k[4]
		# print(label.shape)
		path_ = args.output_dir + '/' + name[0:-4]+'.h5'
		#print path_
		hf = h5py.File(path_, 'w')
		hf.create_dataset('/pose_data', data=pose)
		hf.create_dataset('/orig_data', data=orig)
		hf.create_dataset('/conf', data=conf)
		hf.create_dataset('/label', data=label)
		a_path = os.path.abspath(path_)
		print(a_path)
		txt_file.write(args.output_dir + '/' + name[0:-4]+'.h5\n')
		
		counter = counter + 1	

	print counter
	print len(a.test_data_list)
