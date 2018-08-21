from Framework import postprocessing
from Framework import iterative_gradient_desc

# method that computes the average and variance of a list of numbers
def mean_var(list):
    mean = 0
    n = float(len(list) - 1)
    if n == 0:
        n = 1.0

    for j in list:
        mean += j / n

    var = 0
    for j in list:
        var += ((j - mean) ** 2) / n

    return mean, var

# compares the models in the model_list. Loops multiplies the usual batch size by the defined factor.
def model_comparison(model_list, loops):
    raw_results = {}
    for k in range(loops):
        data = model_list[0]()
        y, x_true, fbp, annos, ul_nod, ul_rand = data.generate_training_data(data.batch_size,
                                                                             noise_level=data.noise_level,
                                                                             scaled=data.scaled)
        for model in model_list:
            recon = model()
            name = recon.model_name + '_' + recon.experiment_name
            ce, mse, total = model.evaluate(y, x_true, fbp, annos, ul_nod, ul_rand)
            if k == 0:
                raw_results[name] = [[], []]
            raw_results[name][0].append(ce)
            raw_results[name][1].append(mse)
            recon.end()

    results = {}
    for name, res in raw_results.items():
        ce = mean_var(res[0])
        mse = mean_var(res[1])
        results[name] = (ce, mse)
        print('Model: {}, CE: {}, MSE: {}'.format(name, ce, mse))

    return results


# run the experiments
n = input('Number of experiment')

# noise = 0.02
noise = 0.05

# model = iterative_gradient_desc
model = postprocessing

# list of experiments run
class Exp1(model):
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

list_1 = [Exp1, Exp1_1, Exp1_2]

# List of experiments with different values of weighting parameter C
list_c = [0.99, 0.9, 0.5, 0.1, 0.01]
list_2 = []
for c in list_c:
    class E(Exp1):
        experiment_name = 'Joint_Weight_' + str(c)
        alpha = c
    list_2.append(E)


if n == 0:
    net = Exp1()
    for k in range(20):
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

    for exp in list_2:
        net = exp()
        for k in range(10):
            net.joint_training(500)
        net.end()

if n == 1:
    model_comparison(model_list=list_1+list_2, loops=10)









