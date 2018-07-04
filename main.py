from Framework import postprocessing



n = input('Number of experiment')


# list of experiments run
class Exp1(postprocessing):
    experiment_name = 'default_experiment'
    channels = 6
    scaled = True
    learning_rate = 0.0001
    batch_size = 4
    noise_level = 0.02


if n == 0:
    net = Exp1()
    for k in range(10):
        net.pretrain_segmentation_true_input(500)

class Exp2(postprocessing):
    experiment_name = 'default_experiment'
    channels = 2
    scaled = False
    learning_rate = 0.0003
    batch_size = 4
    noise_level = 2

if n == 1:
    net = Exp2()
    for k in range(10):
        net.pretrain_segmentation_true_input(500)





