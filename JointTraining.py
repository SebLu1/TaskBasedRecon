from Framework import joint_training

jt = joint_training()
for k in range(3):
    jt.pretrain_reconstruction(200)