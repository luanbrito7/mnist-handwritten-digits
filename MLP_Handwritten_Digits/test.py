# ---------------------
# - Load data:

import data_loader
training_data, validation_data, test_data = data_loader.load_data_wrapper()
training_data = list(training_data)

# ---------------------
# - mlp.py example:
import mlp, time

currentTime = time.time()
net = mlp.MLP([784, 50, 50, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
print("--- %s seconds ---" % (time.time() - currentTime))
