__author__ = 'Lingyan_Wang'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import math
def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(200, noise=0.20)
    y = np.reshape(y,(y.size,1))
    y2 = 1-y
    y = np.concatenate((y,y2),axis=1)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    #print(X)
    #print(y)
    plt.scatter(X[:, 0], X[:, 1], c=y[:,1], cmap=plt.cm.Spectral)
    plt.show()

class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='ReLu', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''
        
        if type == 'tanh':
            activations = np.tanh(z)
        elif type == 'Sigmoid':
            activations = 1 / (1 + math.e ** -z)
        elif type == 'ReLu':
            activations = np.maximum(z,0)
        else:
            print("Not a correct type dd!")
        return activations

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        if type == 'tanh':
            deriv = 1.0 - np.tanh(z)**2
        elif type == 'Sigmoid':
            deriv = 1 / (1 + math.e ** -z)*(1-(1 / (1 + math.e ** -z)))
        elif type == 'ReLu':
            deriv = np.zeros(z.shape)
            deriv[z<=0] = 0;
            deriv[z>0] = 1; 
        else:
            print("Not a correct type!")

        return deriv

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''
   
        self.z1 = X@self.W1 + self.b1
        self.a1 = self.actFun(self.z1,self.actFun_type)
        #print(self.a1.shape)
        self.z2 = self.a1@self.W2 + self.b2
        self.probs = np.exp(self.z2) / np.reshape(np.sum(np.exp(self.z2),1),(np.sum(np.exp(self.z2),1).size,1)) 
        
        return None

    def calculate_loss(self, X, y): # no change 
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        data_loss = -np.sum(np.sum(y*np.log(self.probs)))

        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        print(data_loss)
        return (1. / num_examples) * data_loss

    def predict(self, X): # no change 
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # dL/dW1
        dW2 = (self.a1.T)@(self.probs - y)
        # dL/db2
        db2 = np.ones((1,200))@(self.probs - y)
        # dL/dW1
        mid1 = self.diff_actFun(self.z1,self.actFun_type)
        dW1 = X.T@((self.probs - y)@self.W2.T*mid1)
        # dL/db1
        db1 = np.ones((1,200))@((self.probs - y)@self.W2.T*self.diff_actFun(self.z1,self.actFun_type))
        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)
        
####################################################################################################################################
#################################################################################################################################
#################################################################################################################################

class DeepNeuralNetwork(NeuralNetwork):  
    
        def __init__(self, layer_num, layer_size, actFun_type = 'tanh', reg_lambda=0.01, seed=0):
            '''
            :param nn_input_dim: input dimension
            :param nn_hidden_dim: the number of hidden units
            :param nn_output_dim: output dimension
            :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
            :param reg_lambda: regularization coefficient
            :param seed: random seed
            '''
            self.layer_num = layer_num
            self.layer_size = layer_size
            self.actFun_type = actFun_type
            self.reg_lambda = reg_lambda

            # initialize the weights and biases in the network
            np.random.seed(seed)
            self.W = [];
            self.b = [];

            for i in range(self.layer_size.size-1):
                w_temp = np.random.randn(self.layer_size[i], self.layer_size[i+1])/ np.sqrt(self.layer_size[i])
                b_temp = np.zeros((1, self.layer_size[i+1]))
                self.W.append(w_temp)
                self.b.append(b_temp)


        def feedforward(self, X, actFun):
            '''
            feedforward builds a 3-layer neural network and computes the two probabilities,
            one for class 0 and one for class 1
            :param X: input data
            :param actFun: activation function
            :return:
            '''
            self.z = [] # all the z is calculated with normal feedforward implementation 
            self.a = [] # all but one is calculated with actfun 
            z0 = X@self.W[0] + self.b[0]
            self.z.append(z0)
            a0 = self.actFun(self.z[0],self.actFun_type)
            self.a.append(a0)

            for i in range(self.layer_num-2):
                z_temp = self.a[-1]@self.W[i+1] + self.b[i+1]
                a_temp = self.actFun(z_temp,self.actFun_type)
                self.z.append(z_temp)
                self.a.append(a_temp)

            self.probs = np.exp(self.z[-1]) / np.reshape(np.sum(np.exp(self.z[-1]),1),(np.sum(np.exp(self.z[-1]),1).size,1)) 
            return None

        def backprop(self, X, y):
            '''
            backprop run backpropagation to compute the gradients used to update the parameters in the backward step
            :param X: input data
            :param y: given labels
            :return: dL/dW1, dL/b1, dL/dW2, dL/db2
            '''
            dz = self.probs - y 
            bp = dz
            dW = [];
            db = [];
            for i in range(self.layer_num-2):
                id = self.layer_num - 2 - i
                dW_temp = self.a[id-1].T @ bp
                db_temp = np.ones((1,X.shape[0])) @ bp
                bp = bp @ self.W[id].T*self.diff_actFun(self.z[id-1],self.actFun_type)
                dW.insert(0,dW_temp)
                db.insert(0,db_temp)

            dW.insert(0,X.T @ bp)
            db.insert(0,np.ones((1,X.shape[0])) @ bp)
    
            return dW, db

        def calculate_loss(self, X, y): # no change 
            '''
            calculate_loss compute the loss for prediction
            :param X: input data
            :param y: given labels
            :return: the loss for prediction
            '''
            num_examples = len(X)
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Calculating the loss

            data_loss = -np.sum(np.sum(y*np.log(self.probs)))

            # Add regulatization term to loss (optional)
            for i in range(self.layer_num-1):
                data_loss += self.reg_lambda / 2 * np.sum(np.square(self.W[i]))                

            return (1. / num_examples) * data_loss

        def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
            '''
            fit_model uses backpropagation to train the network
            :param X: input data
            :param y: given labels
            :param num_passes: the number of times that the algorithm runs through the whole dataset
            :param print_loss: print the loss or not
            :return:
            '''
            # Gradient descent.
            for i in range(0, num_passes):
                # Forward propagation
                self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
                # Backpropagation
                dW, db = self.backprop(X, y)
                # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
                for j in range(self.layer_size.size-1):
                    dW[j] += self.reg_lambda * self.W[j]
                    self.W[j] += -epsilon * dW[j]
                    self.b[j] += -epsilon * db[j]
                    
                #if print_loss and i % 1000 == 0:

                    #print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

        
        
def main(ty , num, lsize):
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    #plt.show()
    model = DeepNeuralNetwork(actFun_type = ty, layer_num = num, layer_size = lsize)
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)

#if __name__ == "__main__":
     #main()
