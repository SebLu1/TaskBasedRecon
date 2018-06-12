from Framework import joint_training

jt = joint_training()
# for k in range(3):
#     jt.pretrain_reconstruction(200)

for k in range(5):
    jt.pretrain_segmentation_true_input(500)