import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchsummary import summary as summary_
import pkbar

import warnings
warnings.filterwarnings('ignore')


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # ReLu layer
        self.relu = nn.ReLU()

        # flatten layer
        self.flatten = nn.Flatten()

        # sigmoid layer
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_().to(device)

        # One time step
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.relu(out)
        out = self.flatten(out)
        out = self.sig(out)
        return out


def Input(shape):
    Input.shape = shape
    return Input.shape

# utility function to get CNN output shape


def get_conv_output(shape, inputs):
    bs = 1
    data = Variable(torch.rand(bs, *shape))
    output_feat = inputs(data)

    return output_feat.size(1)

# dense layer


class Dense(nn.Module):
    def __init__(self, outputs, activation):
        super().__init__()
        self.outputs = outputs
        self.activation = activation

    def __call__(self, inputs):
        self.inputs_size = 1

        # if the previous layer is Input layer
        if type(inputs) == tuple:
            for i in range(len(inputs)):
                self.inputs_size *= inputs[i]

            self.layers = nn.Sequential(
                nn.Linear(self.inputs_size, self.outputs),
                self.activation
            )

            return self.layers

        # if the previous layer is dense layer
        elif isinstance(inputs[-2], nn.Linear):
            self.inputs = inputs
            self.layers = list(self.inputs)
            self.layers.extend(
                [nn.Linear(self.layers[-2].out_features, self.outputs), self.activation])

            self.layers = nn.Sequential(*self.layers)

            return self.layers

        # if the previous layer is convolutional layer
        else:
            self.inputs = inputs
            self.layers = list(self.inputs)
            self.layers.extend([nn.Linear(get_conv_output(
                Input.shape, self.inputs), self.outputs), self.activation])

            self.layers = nn.Sequential(*self.layers)

            return self.layers


# custom layer like nn.Linear


class FlattenedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input):
        self.inputs = input.view(input.size(0), -1)
        return self.inputs


# extends the previous layers with FlattenedLayer()


class Flatten():
    def __init__(self):
        pass

    def __call__(self, inputs):
        self.inputs = inputs
        self.layers = list(self.inputs)
        self.layers.extend([FlattenedLayer()])
        self.layers = nn.Sequential(*self.layers)

        return self.layers


# utility fuction


def same_pad(h_in, kernal, stride, dilation):
    return (stride*(h_in-1)-h_in+(dilation*(kernal-1))+1) / 2.0

# Conv2d layer


class Conv2d(nn.Module):
    def __init__(self, filters, kernel_size, strides, padding, dilation, activation):
        super().__init__()
        self.filters = filters
        self.kernel = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.activation = activation

    def __call__(self, inputs):

        if type(inputs) == tuple:
            self.inputs_size = inputs

            if self.padding == 'same':
                self.padding = int(
                    same_pad(self.inputs_size[-2], self.kernel, self.strides, self.dilation))
            else:
                self.padding = self.padding

            self.layers = nn.Sequential(
                nn.Conv2d(self.inputs_size[-3],
                          self.filters,
                          self.kernel,
                          self.strides,
                          self.padding,
                          self.dilation),
                self.activation
            )

            return self.layers

        else:
            if self.padding == 'same':
                self.padding = int(same_pad(get_conv_output(
                    Input.shape, inputs), self.kernel, self.strides, self.dilation))
            else:
                self.padding = self.padding

            self.inputs = inputs
            self.layers = list(self.inputs)
            self.layers.extend(
                [nn.Conv2d(self.layers[-2].out_channels,
                           self.filters,
                           self.kernel,
                           self.strides,
                           self.padding,
                           self.dilation),
                 self.activation]
            )
            self.layers = nn.Sequential(*self.layers)

            return self.layers


class Model():
    def __init__(self, inputs, outputs, device):
        self.input_size = inputs
        self.device = device
        self.model = outputs.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def compile(self, optimizer, loss):
        self.opt = optimizer
        self.criterion = loss

    def summary(self):
        summary_(self.model, self.input_size, device=self.device)
        print("Device Type:", self.device)

    def fit(self, data_x, data_y, epochs):
        self.model.train()

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            progress = pkbar.Kbar(target=len(data_x), width=25)

        for i, (data, target) in enumerate(zip(data_x, data_y)):
            self.opt.zero_grad()

            train_out = self.model(data.to(self.device))
            loss = self.criterion(train_out, target.to(self.device))
            loss.backward()

            self.opt.step()

            progress.update(i, values=[("loss: ", loss.item())])

        progress.add(1)

    def evaluate(self, test_x, test_y):
        self.model.eval()
        correct, loss = 0.0, 0.0

        progress = pkbar.Kbar(target=len(test_x), width=25)

        for i, (data, target) in enumerate(zip(test_x, test_y)):
            out = self.model(data.to(self.device))
            loss += self.criterion(out, target.to(self.device))

            correct += ((torch.max(out, 1)[1]) == target.to(self.device)).sum()

            progress.update(i, values=[
                            ("loss", loss.item()/len(test_x)), ("acc", (correct/len(test_x)).item())])
        progress.add(1)

    def fit_generator(self, generator, epochs):
        self.model.train()

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            progress = pkbar.Kbar(target=len(generator), width=25)

        for i, (data, target) in enumerate(generator):
            self.opt.zero_grad()

            train_out = self.model(data.to(self.device))
            loss = self.criterion(train_out.squeeze(), target.to(self.device))
            loss.backward()

            self.opt.step()

            progress.update(i, values=[("loss: ", loss.item())])

        progress.add(1)

    def evaluate_generator(self, generator):
        self.model.eval()
        correct, loss = 0.0, 0.0

        progress = pkbar.Kbar(target=len(generator), width=25)

        for i, (data, target) in enumerate(generator):
            out = self.model(data.to(self.device))
            loss += self.criterion(out.squeeze(), target.to(self.device))

            correct += (torch.max(out.squeeze(), 1)
                        [1] == target.to(self.device)).sum()

            progress.update(i, values=[
                            ("test_acc", (correct/len(generator)).item()), ("test_loss", loss.item()/len(generator))])

        progress.add(1)

    def predict_generator(self, generator):
        self.model.train()
        out = []
        for i, (data, labels) in enumerate(generator):
            out.append(self.model(data.to(self.device)))

        return out
