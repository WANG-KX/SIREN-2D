import cv2
import torch 
from torch import nn
import numpy as np 
from models import *

SIZE = 256

def pred2img(pred):
    pred = pred.detach().cpu().numpy()
    pred = (pred - pred.min()) / (pred.max()-pred.min())
    pred = (pred*255).astype(np.uint8)
    pred = pred.reshape((SIZE,SIZE))
    return pred

def pred2grad(pred):
    pred = pred.detach().cpu().numpy()
    pred = np.sum(np.abs(pred),axis=1)
    pred = pred - pred.min()
    pred = (pred / pred.max() * 255.0).astype(np.uint8)
    pred = pred.reshape((SIZE,SIZE))
    return pred

image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (SIZE,SIZE))
image_grad = np.zeros((SIZE,SIZE,2))
image_float = image.astype(np.float32)
image_grad[:,1:,0] = (image_float[:,1:] - image_float[:,:-1])
image_grad[1:,:,1] = (image_float[1:,:] - image_float[:-1,:])
image_grad_vis = np.sum(np.abs(image_grad),axis=2)
image_grad_vis = image_grad_vis - image_grad_vis.min() 
image_grad_vis = image_grad_vis/image_grad_vis.max()

cv2.imshow("target", image)
cv2.imshow("target_grad", image_grad_vis)

# generate the target data of training
gradient_torch = torch.from_numpy(image_grad.astype(np.float32)/255.0).cuda()
gradient_torch = gradient_torch.view(-1,2)

# generate input image grids
xs, ys = np.meshgrid(np.linspace(-1,1,SIZE), np.linspace(-1,1,SIZE))
# xs, ys = np.meshgrid(np.linspace(0,511,SIZE), np.linspace(0,511,SIZE))
xs = torch.from_numpy(xs.astype(np.float32)).view(SIZE,SIZE,1)
ys = torch.from_numpy(ys.astype(np.float32)).view(SIZE,SIZE,1)
input_grid = torch.cat([ys, xs],dim=2)
input_grid = input_grid.view(-1,2)
input_grid = input_grid.cuda()

# initialize the models
relu_model = ReLU_Model([2,256,256,256,256,1]).cuda()
relu_pe_model = ReLU_PE_Model([2,256,256,256,256,1], L=5).cuda()
siren_model = SIREN([2,256,256,256,256,1]).cuda()

# initialize the optimizer
parameters = []
parameters += list(relu_model.parameters())
parameters += list(relu_pe_model.parameters())
parameters += list(siren_model.parameters())
optimizer = torch.optim.Adam(parameters, lr=1e-4)

loss_log = []
for iter_idx in range(10000):
    relu_recon = relu_model(input_grid)
    relu_pe_recon = relu_pe_model(input_grid)
    siren_recon = siren_model(input_grid)

    def numerical_grad(pred):
        pred = pred.view(SIZE,SIZE)
        grad = torch.zeros(SIZE,SIZE,2).cuda()
        grad[:,1:,0] = pred[:,1:] - pred[:,:-1]
        grad[1:,:,1] = pred[1:,:] - pred[:-1,:]
        grad = grad.view(-1,2)
        return grad

    relu_grad = numerical_grad(relu_recon)
    relu_pe_grad = numerical_grad(relu_pe_recon)
    siren_grad = numerical_grad(siren_recon)

    relu_loss = torch.mean((relu_grad - gradient_torch)**2)
    relu_pe_loss = torch.mean((relu_pe_grad - gradient_torch)**2)
    siren_loss = torch.mean((siren_grad - gradient_torch)**2)
    total_loss = relu_loss + relu_pe_loss + siren_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    loss_log.append({
        "relu_loss": relu_loss.item(),
        "relu_pe_loss": relu_pe_loss.item(),
        "siren_loss": siren_loss.item()
    })

    if iter_idx % 10 == 0:
        log_str = "relu loss: %f, relu pe loss: %f, siren loss: %f" % (
            relu_loss.item(), relu_pe_loss.item(), siren_loss.item())
        print(log_str)
        relu_img = pred2img(relu_recon)
        relu_pe_img = pred2img(relu_pe_recon)
        siren_img = pred2img(siren_recon)
        big_img = np.concatenate([relu_img, relu_pe_img, siren_img], axis=1)

        relu_grad = pred2grad(relu_grad)
        relu_pe_grad = pred2grad(relu_pe_grad)
        siren_grad = pred2grad(siren_grad)
        big_grad = np.concatenate([relu_grad, relu_pe_grad, siren_grad], axis=1)

        big_img = np.concatenate([big_img, big_grad], axis=0)

        cv2.imshow("reconstructed img", big_img)
        cv2.waitKey(100)
