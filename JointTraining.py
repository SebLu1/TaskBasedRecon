from Framework import joint_training

# jt = joint_training()
#
# for k in range(3):
#     jt.pretrain_reconstruction(200)
#
# for k in range(5):
#     jt.pretrain_segmentation_true_input(500)

class joint_training_seg(joint_training):
    experiment_name = 'Pretrain_Seg_on_Reconstruction'

net = joint_training_seg()
for k in range(5):
    net.pretrain_segmentation_reconstruction_input(steps=200)