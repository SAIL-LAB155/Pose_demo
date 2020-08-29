import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from config.config import device


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, height, width):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.height = height
        self.width = width
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim*self.height*self.width, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        #print(batch_size)
        energy = self.projection(encoder_outputs)
        #print(energy.shape)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        #print(weights.shape)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        if device != "cpu":
            return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                    Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                    Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, num_classes, batch_first=False, bias=True, return_all_layers=False, batch_size=1, attention = None):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.attention = attention
        

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i], 
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.attention_layer = SelfAttention(hidden_dim[-1], self.height, self.width)
        self.output_layer = nn.Sequential(nn.Linear(self.hidden_dim[-1]*self.height*self.width, self.hidden_dim[-1]),
            nn.BatchNorm1d(self.hidden_dim[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dim[-1], self.num_classes))
            



    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]
        
        x = layer_output_list[0][:,-1,:,:,:]
        final_hidden_state = last_state_list[0][0]
        #print('final_hidden_state shape ', final_hidden_state.shape)
        if self.attention:
            x = x.view(x.size(0),-1).unsqueeze(1)
        #print(x.shape)
            embedding, attn_weights = self.attention_layer(x)
            output = self.output_layer(embedding)
        #x = self.output_layer(x)
            return output
        else:
            x = x.view(x.size(0),-1)
            return self.output_layer(x)



    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    model = ConvLSTM(input_size=(17, 2),
                 input_dim=1,
                 hidden_dim=[128, 64, 32],
                 kernel_size=(3, 3),
                 num_layers=3,
                 num_classes=2,
                 batch_size=2,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 attention=False).cuda()

    input_tensor = torch.randn(2,30,1,17,2).cuda()
    target = torch.randn(2,30,32,17,2).cuda()
    output = model(input_tensor)
    print(output.shape)
