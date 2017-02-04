# Extract Features Using PoseBox-re-id

1. Download res50-TripleLoss-dropout__iter_40000.caffemodel / res50-TripleLoss__iter_40000.caffemodel and save in ../Models folder

   detail see ../Models/download-list

2. Run the python script to extract features:

   1) The script will dump four kind of features:

					1] orig-fc751: the output of final fc layer with 751 classes from original image datastream

					2] pose-fc751: the output of final fc layer with 751 classes from posebox image datastream

					3] overall-fc751: the output of final fc layer 751 classes from the merged two datastreams

					4] concat: the concate 4096-dim features from combination of two last pooling layers in two datastream

   2) Create Posebox Data for your data:

      Please fellow the instruction in ../DataMaker to generate correct poseBox and the .csv file that records the confidence scores of pose estimation.

   2) Example:

      python feature_extractor.py
				  --caffemodel ../Models/res50-TripleLoss-dropout__iter_40000.caffemodel

      	     			  --orig_data /data/data/Market-1501-v15.09.15/pose_box_train

				  --pose_data /data/data/Market-1501-v15.09.15/bounding_box_train

				  --csv_list ../DataMaker/re-id_pose.csv