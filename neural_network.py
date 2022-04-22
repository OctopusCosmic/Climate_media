import numpy as np
import matplotlib.pyplot as plt

def main():
  N = 100 # number of points per class
  D = 2 # dimensionality
  K = 3 # number of classes
  X = np.zeros((N*K,D)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels
  for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j


  # lets visualize the data:
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  plt.show()

  #### Train a Neural Network Classifier ####/*

  # initialize parameters randomly
  h = 100 # size of hidden layer
  W = 0.01 * np.random.randn(D,h)
  b = np.zeros((1,h))
  W2 = 0.01 * np.random.randn(h,K)
  b2 = np.zeros((1,K))

  # some hyperparameters
  step_size = 1e-0
  reg = 1e-3 # regularization strength

  # gradient descent loop
  num_examples = X.shape[0]
  for i in range(10000):

    # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if i % 1000 == 0:
      print("iteration %d: loss %f" % (i, loss))

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW += reg * W

    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

  # evaluate training set accuracy
  hidden_layer = np.maximum(0, np.dot(X, W) + b)
  scores = np.dot(hidden_layer, W2) + b2
  predicted_class = np.argmax(scores, axis=1)
  print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
  #### Train a Neural Network Classifier ####*/

  class NeuralNet(nn.Module):
    def __init__(self):
      super(NeuralNet, self).__init__()
      self.flatten = nn.Flatten()
      self.linear_relu_stack = nn.Sequential(
        nn.Linear(100, 80), # 80 is size of hidden layer
        nn.ReLU(),
        nn.Linear(80, 18), # tuning this
      )

    def forward(self, x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return


def train_NN_option1(whole_context, whole_tag_loc, dimension):
  epochs = 450
  batch_size_original = 64
  learning_rate = 0.01
  dimension = dimension * 3
  nnet = NeuralNet()
  grad_desc = optim.SGD(nnet.parameters(), lr=learning_rate)
  loss_func = nn.CrossEntropyLoss()
  for e in range(epochs):
    size = batch_size_original
    for i in range(0, len(whole_context), size):
      if i + size > len(whole_context):
        size = len(whole_context) - i
      grad_desc.zero_grad()
      data = autograd.Variable(whole_context[i:(i + size)].data.view(size, dimension), requires_grad=True)
      predictions = nnet(data)
      labels = autograd.Variable(whole_tag_loc[i:(i + size)].data)
      loss = loss_func(predictions, labels)
      loss.backward(retain_graph=True)
      grad_desc.step()
  return nnet