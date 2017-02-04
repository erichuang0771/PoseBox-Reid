# PoseBox-Reid

Pedestrian misalignment, which mainly arises from detector errors and pose variations, is a critical problem for a robust person re-identification (re-ID) system. With bad alignment, the background noise will significantly compromise the feature learning and matching process. To address this problem, this paper introduces the pose invariant embedding (PIE) as a pedestrian descriptor. First, in order to align pedestrians to a standard pose, the PoseBox structure is introduced, which is generated through pose estimation followed by affine transformations. Second, to reduce the impact of pose estimation errors and information loss during PoseBox construction, we design a PoseBox fusion (PBF) CNN architecture that takes the original image, the PoseBox, and the pose estimation confidence as input. The proposed PIE descriptor is thus defined as the fully connected layer of the PBF network for the retrieval task. Experiments are conducted on the Market-1501, CUHK03, and VIPeR datasets. We show that PoseBox alone yields decent re-ID accuracy and that when integrated in the PBF network, the learned PIE descriptor produces competitive performance compared with the state-of-the-art approaches. 



Paper: https://arxiv.org/abs/1701.07732

