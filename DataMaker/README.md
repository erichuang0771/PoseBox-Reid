# Data Preparation

There are main three steps to create data for PoseBox-Reid:


## 1) Extract pose information from original image:
    Here we are using "Convolutional Pose Machines" to extarct pose information from orginal data, see: https://arxiv.org/abs/1602.00134
    
    Here we prvoide a Torch version of "Convolutional Pose Machines" to estimate pose, but feel free to replace this part with another method

    *** Pre-request:
        1. torchx
        2. torch-opencv
    
    1. Download TPM_pose_estimation_lite.t7 and save in ../Models folder, see ../Models/download_list
    
    2. Modify Data folder Path in run_CPM.lua

       Ex: -- Modify Here to change data dir:
        data_dir = '/data/data/Market-1501-v15.09.15/bounding_box_train/'
    
    3. Run Torch Convolution Pose Machines: 

       Ex: th run_CPM.lua
    
    4. The Program will generate a "re-id_pose.csv" file that records pose estimation results 
    
## 2) Create Posebox from original data:
   Here we provide Matlab scripts (in poseBox_Maker folder) used to generate Posebox Data based on the Pose estimation results
   
   1. check gen_posebox.m to redirect input and output to our data folder

      WARNING: it's recommand to save "re-id_pose.csv" as .xls format, the xlsread() function may vary depends on your OS.
   
   2. Run scipts to generate all poseBox based on the pose estimation results:

      Ex: matlab ./gen_posebox.m

## 3) Create hdf5 database (for training only):
 
    1. Check hdf5maker.py to build hdf5 database for training.

    Ex:  python hdf5maker.py --orig_data /data/data/Market-1501-v15.09.15/bounding_box_train 
                                   --pose_data /data/data/Market-1501-v15.09.15/pose_box_train 
                                   --csv_list re-id_pose.csv 
                                   --output_dir /data/data/Market_hdf5
                                   
