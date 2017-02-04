
To Download the Models:

visit: www.visualsolver.com:6008

Five models are availble:
1)  ResNet_OrigImg__iter_40000.caffemodel
    Resnet 50 trained on original Market Dataset (90% training & 10% validation)

2) ResNet_PoseBox__iter_40000.caffemodel
   Resnet 50 trained on posebox data made from Market Dataset (90% training & 10% validation)

3) res50-TripleLoss2-init.caffemodel
   Two stream net merged two from above two network

4) res50-TripleLoss__iter_40000.caffemodel
   Two stream net above finetuned on Martet Dataset with TripleLoss

5) res50-TripleLoss-dropout__iter_40000.caffemodel
   Above model trained with dropout layer

Also:

1) A CPM torch verion model avaible to output pose info:
   TPM_pose_estimation_lite.t7
