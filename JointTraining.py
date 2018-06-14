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


if n ==3:
    test_size = 100

    class joint_training_seg(joint_training):
        experiment_name = 'Pretrain_Seg_on_Reconstruction'

    net = joint_training_seg()
    ce1, ce2, ce_total, l2 = net.evaluate(test_size=test_size)
    print(ce1)
    print(ce2)
    print(ce_total)
    print(l2)



def comparison():
    jt = joint_training()
    y, x_true, fbp, annos, ul_nod, ul_rand = jt.generate_training_data(jt.batch_size, training_data=False,
                                                                       noise_level=0.02)
    recon = []
    nod = []
    anno = []
    seg = []
    r, n, anno, seg = jt.compute(y, x_true, fbp, annos, ul_nod, ul_rand)
    jt.end()


