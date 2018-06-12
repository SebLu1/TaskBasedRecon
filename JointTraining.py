from Framework import joint_training

jt = joint_training()
for k in range(5):
    jt.train(200)