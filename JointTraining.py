from Framework import joint_training



n = input('Number of experiment')


class joint_training_seg(joint_training):
    experiment_name = 'Pretrain_Seg_on_Reconstruction'

class fully_joint_training(joint_training):
    experiment_name = 'Joint_Training'


if n==0:

    jt = joint_training()
    for k in range(5):
        jt.pretrain_segmentation_true_input(500)

if n==1:

    net = joint_training_seg()
    for k in range(5):
        net.pretrain_segmentation_reconstruction_input(steps=200)

if n ==2:

    net = fully_joint_training()
    print('Weighting parameter alpha:' + str(net.alpha))
    for k in range(5):
        net.joint_training(steps=200)


if n ==3:
    test_size = 100


    net = joint_training()
    ce1, ce2, ce_total, l2 = net.evaluate(test_size=test_size, direct_feed=True)
    print(net.experiment_name)
    print(ce1)
    print(ce2)
    print(ce_total)
    print(l2)
    net.end()

    net = joint_training_seg()
    ce1, ce2, ce_total, l2 = net.evaluate(test_size=test_size)
    print(net.experiment_name)
    print(ce1)
    print(ce2)
    print(ce_total)
    print(l2)
    net.end()

    net = fully_joint_training()
    ce1, ce2, ce_total, l2 = net.evaluate(test_size=test_size)
    print(net.experiment_name)
    print(ce1)
    print(ce2)
    print(ce_total)
    print(l2)
    net.end()

### massive joint run
if n ==4:
    jt = joint_training()
    for k in range(10):
        jt.pretrain_segmentation_true_input(500)
    jt.end()

### use malignancy prediction as well
from Framework import joint_training_mal






