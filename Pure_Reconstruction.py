from Framework import postprocessing

pp = postprocessing()
for k in range(5):
    pp.train(200)