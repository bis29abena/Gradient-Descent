# Usage
# python main.py

# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


# compute the sigmoid activation value for a given inpu
def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


# compute the derivative of the sigmoid fucntion ASSUMING
# that the input 'x' has already passed through the sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)


def predict(X, W):
    # take the dot product between our features and weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply a step function to threshold the output to binary class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    # return predictions
    return preds


# construct an argparse to parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="number of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1000 data points, where
# each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a col of 1s at the last entry in the feature matrix
# this little trick allows us to treat the bias as a trainable
# parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% for both
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    # take the dot product between our features X and the weight matrix W, then pass this value through
    # our sigmoid activation function, thereby giving us our predictions on the dataset
    preds = sigmoid_activation(trainX.dot(W))

    # now that we have our predictions, we need to determine the error, which is the difference
    # between our predictions and the true value
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # the gradient descent update is the dot product between our features and the error of the sigmoid activation
    # of our predictions.
    d = error * sigmoid_derivative(preds)
    gradient = trainX.T.dot(d)

    # in the update stage, all we need to do is nudge the weight matrix in the negative direction of the gradient
    # (hence the term gradient descent by taking a small step towards a set of more optimal parameters)
    W += -args["alpha"] * gradient

    # check to see if any update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"[INFO] epoch={int(epoch + 1)}, loss={loss:.7f}")

# evaluate our model
print("[INFO] evaluating....")
preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)
plt.savefig("Data")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.savefig("Training Loss")
plt.show()