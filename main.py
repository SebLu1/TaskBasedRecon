from Framework import postprocessing



n = input('Number of experiment')


# list of experiments run
class exp1(postprocessing):
    experiment_name = 'default_experiment'
    channels = 6
    scaled = True


if n == 0:
    net = exp1()
    for k in range(5):
        net.pretrain_segmentation_true_input(100)







