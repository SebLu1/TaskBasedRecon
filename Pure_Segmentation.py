from Framework import pure_segmentation

ps = pure_segmentation()
for k in range(5):
    ps.train(200)