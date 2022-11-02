import math
import torch
import torchvision
import torch.nn as nn
from torch import nn
from torch.autograd import Variable
# from torchsummary import summary as summary
from typing import Callable
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layer_dim: int,
        bidirectional=False,
    ) -> None:
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # Output dim classes
        self.output_dim = output_dim
        self.n_directions = 2 if bidirectional else 1

        # LSTM Layer
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim,
                                  batch_first=True, bidirectional=bidirectional)
        # Linear Dense
        self.fc1 = torch.nn.Linear(hidden_dim*self.n_directions, hidden_dim//2)
        # Linear Dense
        self.fc2 = torch.nn.Linear(hidden_dim//2, self.output_dim)
        # initialize weights & bias with stdv -> 0.05
        self.initialization()

    def init_hidden(self, x: torch.Tensor) -> torch.FloatTensor:
        h0 = torch.zeros(
            self.layer_dim*self.n_directions,
            x.size(0),
            self.hidden_dim,
            device="cuda"
        ).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(
            self.layer_dim*self.n_directions,
            x.size(0),
            self.hidden_dim,
            device="cuda"
        ).requires_grad_()
        return h0, c0

    def forward(self, x) -> torch.Tensor:
        # One time step
        out, _ = self.lstm(x, self.init_hidden(x))
        # Dense lstm
        out = self.fc1(out)
        # Dense for softmax
        out = self.fc2(out)
        return out[:, -1]

    def initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(param.data, std=0.05)
            elif isinstance(m, torch.nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.xavier_uniform_(
                                param[i*mul:(i+1)*mul])
                    elif 'weight_hh' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.xavier_uniform_(
                                param[i*mul:(i+1)*mul])
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param.data)


class CNN_LSTM(nn.Module):
    def __init__(
        self,
        cnn: nn.Module,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layer_dim: int,
        bidirectional=False,
    ) -> None:
        super(CNN_LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # Output dim classes
        self.output_dim = output_dim
        self.n_directions = 2 if bidirectional else 1

        # Base cnn features layer
        self.extractor = cnn.features
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim,
                            batch_first=True, bidirectional=bidirectional)
        # Linear Dense
        self.fc1 = nn.Linear(hidden_dim*self.n_directions, hidden_dim//2)
        # Linear Dense
        self.fc2 = nn.Linear(hidden_dim//2, self.output_dim)
        # initialize weights & bias with stdv -> 0.05
        self.initialization()

    def init_hidden(self, x: torch.Tensor) -> torch.FloatTensor:
        h0 = torch.zeros(
            self.layer_dim*self.n_directions,
            x.size(0),
            self.hidden_dim,
            device="cuda"
        ).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(
            self.layer_dim*self.n_directions,
            x.size(0),
            self.hidden_dim,
            device="cuda"
        ).requires_grad_()
        return h0, c0

    def forward(self, x) -> torch.Tensor:
        batch_size, chunk_size = x.shape[:2]
        # Get features (4,10,3,360,360) -> (40,3,360,360)
        out = self.extractor(x.flatten(end_dim=1)).flatten(start_dim=1)
        # Shape back to batch x chunk
        out = out.reshape((batch_size, chunk_size, out.shape[-1]))
        # One time step
        out, self.hidden_state = self.lstm(out, self.init_hidden(x))
        # Dense lstm
        out = self.fc1(out)
        # Dense for softmax
        out = self.fc2(out)
        return out[:, -1]

    def initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(param.data, std=0.05)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            nn.init.xavier_uniform_(param[i*mul:(i+1)*mul])
                    elif 'weight_hh' in name:
                        for i in range(4):
                            mul = param.shape[0]//4
                            nn.init.xavier_uniform_(param[i*mul:(i+1)*mul])
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)


class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:

    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        d_model: int = 512,
        batch_first: bool = False
    ):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):

    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".
    A detailed description of the code can be found in my article here:
    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.
    Unlike the paper, this class assumes that input layers, positional encoding 
    layers and linear mapping layers are separate from the encoder and decoder, 
    i.e. the encoder and decoder only do what is depicted as their sub-layers 
    in the paper. For practical purposes, this assumption does not make a 
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the 
    Encoder() and Decoder() classes.
    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020). 
    'Deep Transformer Models for Time Series Forecasting: 
    The Influenza Prevalence Case'. 
    arXiv:2001.08317 [cs, stat] [Preprint]. 
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).
    [2] Vaswani, A. et al. (2017) 
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint]. 
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).
    """

    def __init__(self,
                 input_size: int,
                 dec_seq_len: int,
                 batch_first: bool,
                 out_seq_len: int = 58,
                 dim_val: int = 512,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dropout_decoder: float = 0.2,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 2048,
                 dim_feedforward_decoder: int = 2048,
                 num_predicted_features: int = 1
                 ):
        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder
            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__()

        self.dec_seq_len = dec_seq_len

        #print("input_size is: {}".format(input_size))
        #print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
        )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
        )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
        )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]

        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """

        #print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        #print("From model.forward(): tgt size = {}".format(tgt.size()))

        # Pass throguh the input layer right before the encoder
        # src shape: [batch_size, src length, dim_val] regardless of number of input features
        src = self.encoder_input_layer(src)
        #print("From model.forward(): Size of src after input layer: {}".format(src.size()))

        # Pass through the positional encoding layer
        # src shape: [batch_size, src length, dim_val] regardless of number of input features
        src = self.positional_encoding_layer(src)
        #print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length.
        # (https://github.com/huggingface/transformers/issues/4083)
        src = self.encoder(  # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
        )
        #print("From model.forward(): Size of src after encoder: {}".format(src.size()))

        # Pass decoder input through decoder input layer
        # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
        decoder_output = self.decoder_input_layer(tgt)
        #print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))

        # if src_mask is not None:
        #print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        # if tgt_mask is not None:
        #print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )

        #print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

        # Pass through linear mapping
        # shape [batch_size, target seq len]
        decoder_output = self.linear_mapping(decoder_output)
        #print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))

        return decoder_output


"""
https://stackoverflow.com/questions/53628622/loss-function-its-inputs-for-binary-classification-pytorch
https://towardsdatascience.com/recreating-keras-functional-api-with-pytorch-cc2974f7143c
"""


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

    # def summary(self):
    #     summary(self.model, self.input_size, device=self.device)
    #     print("Device Type:", self.device)

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
