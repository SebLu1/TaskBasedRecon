from Framework import postprocessing



n = input('Number of experiment')


# noise = 0.02
noise = 0.05

# list of experiments run
class Exp1(postprocessing):
    experiment_name = 'default_experiment'
    channels = 6
    scaled = True
    learning_rate = 0.0001
    batch_size = 4
    noise_level = noise

class Exp1_1(Exp1):
    experiment_name = 'SegmentationTrainedOnly'

class Exp1_2(Exp1):
    experiment_name = 'JointTraining'


if n == 0:
    net = Exp1()
    for k in range(10):
        net.pretrain_segmentation_true_input(500)
    net.end()

    net = Exp1()
    for k in range(10):
        net.pretrain_reconstruction(500)
    net.end()

    net = Exp1_1()
    for k in range(10):
        net.pretrain_segmentation_reconstruction_input(500)
    net.end()

    net = Exp1_2()
    for k in range(10):
        net.joint_training(500)
    net.end()

# method that computes the average and variance of a list of numbers
def mean_var(list):
    mean  = 0
    n = float(len(list)-1)
    if n == 0:
        n = 1.0

    for j in list:
        mean += j/n

    var = 0
    for j in list:
        var += ((j - mean)**2 )/n

    return mean, var

# compares the models performance
def model_comparison(model_list, iterations):
    data = model_list[0]()
    y, x_true, fbp, annos, ul_nod, ul_rand = data.generate_training_data(data.batch_size, noise_level=data.noise_level,
                                                                         scaled=data.scaled)

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





