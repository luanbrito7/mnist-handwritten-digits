import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# ---------------------
# - network.py example:
import network, time

currentTime = time.time()
net = network.Network([784, 10, 10])
net.SGD(training_data, 30, 10, 4.0, test_data=test_data)
print("--- %s seconds ---" % (time.time() - currentTime))