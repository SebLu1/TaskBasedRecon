from Framework import postprocessing



n = input('Number of experiment')


noise = 0.02

# list of experiments run
class Exp1(postprocessing):
    experiment_name = 'default_experiment'
    channels = 6
    scaled = True
    learning_rate = 0.0001
    batch_size = 4
    noise_level = noise


if n == 0:
    net = Exp1()
    for k in range(10):
        net.pretrain_segmentation_true_input(500)
    net.end()

if n == 1:
    net = Exp1()
    for k in range(10):
        net.pretrain_reconstruction(500)
    net.end()

class Exp1_1(Exp1):
    experiment_name = 'SegmentationTrainedOnly'

if n == 2:
    net = Exp1_1()
    for k in range(10):
        net.pretrain_segmentation_reconstruction_input(100)
    net.end()

class Exp1_2(Exp1):
    experiment_name = 'JointTraining'

if n == 3:
    net = Exp1_2()
    for k in range(10):
        net.joint_training(100)
    net.end()


### Comparison Experiments without malignancy prediction
class Exp2(postprocessing):
    experiment_name = 'default_experiment'
    channels = 2
    scaled = False
    learning_rate = 0.0003
    batch_size = 4
    noise_level = 2

if n == 10:
    net = Exp2()
    for k in range(10):
        net.pretrain_segmentation_true_input(500)





