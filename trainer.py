import torch
from torch.autograd import Variable
import numpy as np
from collections import defaultdict, OrderedDict

def get_adv_weight(adv_weight, epoch):
    if isinstance(adv_weight, list):
        if epoch < 10:
            return adv_weight[0]
        elif epoch < 30:
            return adv_weight[1]
        elif epoch < 60:
            return adv_weight[2]
        else:
            return adv_weight[3]
    else: # just one number
        return adv_weight

def is_power_two(n):
    mod = np.mod(np.log(n) / np.log(2), 1)
    return mod < 1e-9 or mod > 1 - 1e-9

def train_CE(G_net, D_net, device, criterion_pxl, criterion_D, optimizer_G, optimizer_D,
             data_loaders, model_save_path, html_save_path, n_epochs=200, start_epoch=0, adv_weight=0.001, input_shape=(256,256)):
    '''
    Based on Context Encoder implementation in PyTorch.
    '''
    Tensor = torch.cuda.FloatTensor
    hist_loss = defaultdict(list)
    patch_h = input_shape[0] // 8
    patch_w = input_shape[1] // 8
    patch = (1, patch_h, patch_w)

    for epoch in range(start_epoch, n_epochs):
        for phase in ['train', 'val']: # for each epoch, alternate train and val
            batches_done = 0
            running_loss_pxl = 0.0 # pixel level loss: L1 loss
            running_loss_adv = 0.0
            running_loss_D = 0.0
            for idx, (imgs, masked_imgs, masked_parts) in enumerate(data_loaders[phase]): # for each batch
                if phase == 'train':
                    G_net.train()
                    D_net.train()
                else:
                    G_net.eval()
                    D_net.eval()
                torch.set_grad_enabled(phase == 'train')
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False).to(device)
                # valid = all output features from D are 1
                # patch = (1, patch_h, patch_w), so (imgs.shape[0], *patch)=[B,1,H_out,W_out], H_out=W_out=output_size//8 
                # .fill_() is in-place fill
                # Configure input
                imgs = Variable(imgs.type(Tensor)).to(device)
                masked_imgs = Variable(masked_imgs.type(Tensor)).to(device)
                # -----------
                #  Generator
                # -----------
                if phase == 'train':
                    optimizer_G.zero_grad()
                # Generate a batch of images
                outputs = G_net(masked_imgs)
                # Adversarial and pixelwise loss
                loss_pxl = criterion_pxl(outputs, imgs)
                loss_adv = criterion_D(D_net(outputs), valid) # adv loss for gen: the opinon from D
                # Total loss
                cur_adv_weight = get_adv_weight(adv_weight, epoch)
                loss_G = (1 - cur_adv_weight) * loss_pxl + cur_adv_weight * loss_adv
                if phase == 'train':
                    loss_G.backward()
                    optimizer_G.step()
                # ---------------
                #  Discriminator
                # ---------------
                if phase == 'train':
                    optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = criterion_D(D_net(imgs), valid)
                fake_loss = criterion_D(D_net(outputs.detach()), fake)
                loss_D = 0.5 * (real_loss + fake_loss)
                if phase == 'train':
                    loss_D.backward()
                    optimizer_D.step()
                # Update & print statistics
                batches_done += 1
                running_loss_pxl += loss_pxl.item()
                running_loss_adv += loss_adv.item()
                running_loss_D += loss_D.item()
                if phase == 'train' and is_power_two(batches_done):
                    print('Batch {:d}/{:d}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}'.format(
                          batches_done, len(data_loaders[phase]), loss_pxl.item(), loss_adv.item(), loss_D.item()))
            # Store model & visualize examples
            if phase == 'train':
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(G_net.state_dict(), model_save_path + '/G_' + str(epoch) + '.pt')
                torch.save(D_net.state_dict(), model_save_path + '/D_' + str(epoch) + '.pt')
                # generate_html(G_net, D_net, device, data_loaders, html_save_path + '/' + str(epoch), outpaint=outpaint)
            # Store & print statistics
            cur_loss_pxl = running_loss_pxl / batches_done
            cur_loss_adv = running_loss_adv / batches_done
            cur_loss_D = running_loss_D / batches_done
            hist_loss[phase + '_pxl'].append(cur_loss_pxl)
            hist_loss[phase + '_adv'].append(cur_loss_adv)
            hist_loss[phase + '_D'].append(cur_loss_D)
            print('Epoch {:d}/{:d}  {:s}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}'.format(
                  epoch + 1, n_epochs, phase, cur_loss_pxl, cur_loss_adv, cur_loss_D))
    print('Done!')
    return hist_loss