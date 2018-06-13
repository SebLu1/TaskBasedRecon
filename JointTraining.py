from Framework import joint_training



n = input('Number of experiment')

if n==0:

    jt = joint_training()
    for k in range(5):
        jt.pretrain_segmentation_true_input(500)

if n==1:

    class joint_training_seg(joint_training):
        experiment_name = 'Pretrain_Seg_on_Reconstruction'

    net = joint_training_seg()
    for k in range(5):
        net.pretrain_segmentation_reconstruction_input(steps=200)

if n ==2:

    class fully_joint_training(joint_training):
        experiment_name = 'Joint_Training'

    net = fully_joint_training()
    print('Weighting parameter alpha:' + str(net.alpha))
    for k in range(5):
        net.joint_training(steps=200)