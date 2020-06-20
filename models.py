import torch 
from torch import nn 
import math 

class ReLU_Model(nn.Module):
    '''this is vanila relu fully connected model'''

    def __init__(self, dims):
        super(ReLU_Model, self).__init__()
        # here we use the default initialization
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i],dims[i+1]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, get_gradient=False):
        # save middle result for gradient calculation
        relu_masks = []
        middle_result = x 
        for layer in self.layers[:-1]:
            middle_result = self.relu(layer(middle_result))
            relu_mask = (middle_result > 0)
            relu_mask.type_as(middle_result)
            relu_masks.append(relu_mask)
        # last layer
        result = self.layers[-1](middle_result)

        if not get_gradient:
            return result

        # do backwards 
        B = x.shape[0]
        gradient = self.layers[-1].weight
        gradient = gradient.repeat(B,1)
        for i in range(len(self.layers)-2, -1, -1):
            layer_relu_mask = relu_masks[i]
            layer_gradient_weight = self.layers[i].weight
            gradient = gradient * layer_relu_mask
            gradient = torch.matmul(gradient, layer_gradient_weight)

        return result, gradient 

class ReLU_PE_Model(nn.Module):
    '''this is vanila relu fully connected model with position encoding'''

    def __init__(self, dims, L):
        '''L is the level of position encoding'''
        super(ReLU_PE_Model, self).__init__()
        self.L = L
        dims[0] = dims[0] + dims[0]*2*L
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            # here we use the default initialization
            self.layers.append(nn.Linear(dims[i],dims[i+1]))
        self.relu = nn.ReLU(inplace=True)

    def position_encoding_forward(self,x):
        B,C = x.shape
        x = x.view(B,C,1)
        results = [x]
        for i in range(1, self.L+1):
            freq = (2**i) * math.pi
            cos_x = torch.cos(freq*x)
            sin_x = torch.sin(freq*x)
            results.append(cos_x)
            results.append(sin_x)
        results = torch.cat(results, dim=2)
        results = results.permute(0,2,1)
        results = results.reshape(B,-1)
        return results

    def position_encoding_backward(self,x):
        B,C = x.shape
        x = x.view(B,C,1)
        results = [torch.ones_like(x)]
        for i in range(1, self.L+1):
            freq = (2**i) * math.pi
            cos_x_grad = -1.0*torch.sin(freq*x)*freq
            sin_x_grad = torch.cos(freq*x)*freq
            results.append(cos_x_grad)
            results.append(sin_x_grad)
        results = torch.cat(results, dim=2)
        results = results.permute(0,2,1)
        results = results.reshape(B,-1)
        return results

    def forward(self, x, get_gradient=False):
        # save middle result for gradient calculation
        relu_masks = []
        x_pe = self.position_encoding_forward(x)
        middle_result = x_pe
        for layer in self.layers[:-1]:
            middle_result = self.relu(layer(middle_result))
            relu_mask = (middle_result > 0)
            relu_mask.type_as(middle_result)
            relu_masks.append(relu_mask)
        # last layer
        result = self.layers[-1](middle_result)

        if not get_gradient:
            return result

        # do backwards 
        B,C = x.shape
        gradient = self.layers[-1].weight
        gradient = gradient.repeat(B,1)
        for i in range(len(self.layers)-2, -1, -1):
            layer_relu_mask = relu_masks[i]
            layer_gradient_weight = self.layers[i].weight
            gradient = gradient * layer_relu_mask
            gradient = torch.matmul(gradient, layer_gradient_weight)
        # backward the gradient of position encoding
        pe_gradient = self.position_encoding_backward(x)
        gradient = gradient * pe_gradient
        gradient = gradient.view(B, -1, C)
        gradient = torch.sum(gradient, dim=1, keepdim=False)
        return result, gradient 

class SIREN(nn.Module):
    '''
    this model is a fully connected model with sin active func.
    basic implementation of Implicit Neural Representations with Periodic Activation Functions
    '''
    def __init__(self, dims):
        super(SIREN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i],dims[i+1]))

        # initialize the weights, except the last layer
        for i in range(len(dims)-2):
            in_dim = dims[i]
            bound = math.sqrt(6.0 / in_dim)
            if i == 0:
                bound = bound * 30.0
            nn.init.uniform_(self.layers[i].weight, a=-bound, b=bound)
            nn.init.uniform_(self.layers[i].bias, a=-1.0/in_dim, b=1.0/in_dim)

    def forward(self, x, get_gradient=False):
        # save middle result for gradient calculation
        saved_result = []
        middle_result = x 
        for layer in self.layers[:-1]:
            linear_result = layer(middle_result)
            saved_result.append(linear_result)
            middle_result = torch.sin(linear_result)
        # last layer
        result = self.layers[-1](middle_result)

        if not get_gradient:
            return result

        # do backwards 
        B = x.shape[0]
        gradient = self.layers[-1].weight
        gradient = gradient.repeat(B,1)
        for i in range(len(self.layers)-2, -1, -1):
            layer_gradient_cos = torch.cos(saved_result[i])
            layer_gradient_weight = self.layers[i].weight
            gradient = gradient * layer_gradient_cos
            gradient = torch.matmul(gradient, layer_gradient_weight)

        return result, gradient 

if __name__ == "__main__":
    ## check SIREN gradient calculation
    model = SIREN([2,128,129,1]).double()
    inputs = torch.rand(1000,2).double()
    outputs, gradients = model(inputs, True)
    inputs[:,0] += 1e-8
    outputs2, _ = model(inputs, True)
    numerical_gradient = (outputs2 - outputs) / 1e-8
    gradient_diff = torch.abs(gradients[:,0] - numerical_gradient.squeeze())
    print(gradient_diff.max())

    # ## check the ReLU_Model gradient calculation
    # model = ReLU_Model([2,128,1]).double()
    # inputs = torch.rand(1000,2).double()
    # outputs, gradients = model(inputs, True)
    # inputs[:,0] += 1e-8
    # outputs2, _ = model(inputs, True)
    # numerical_gradient = (outputs2 - outputs) / 1e-8
    # gradient_diff = torch.abs(gradients[:,0] - numerical_gradient.squeeze())
    # print(gradient_diff.max())

    # ## check the ReLU_PE_Model gradient calculation
    # modle = ReLU_PE_Model([2,128,1], L = 5).double()
    # inputs = torch.rand(1000,2).double()
    # outputs, gradients = modle(inputs, True)
    # inputs[:,0] += 1e-8
    # outputs2, _ = modle(inputs, True)
    # numerical_gradient = (outputs2 - outputs) / 1e-8
    # gradient_diff = torch.abs(gradients[:,0] - numerical_gradient.squeeze())
    # print(gradient_diff.max())