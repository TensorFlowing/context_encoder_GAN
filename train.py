import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" # physical indeces of gpu, corresponding to [0,1,2,3] logical indeces
import matplotlib.pyplot as plt

def tensor2np(img):
    img = img.cpu().numpy()
    img = img.squeeze().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img

if __name__ == '__main__':

    import torch
    from data_prep import CEImageDataset
    from model import CEGenerator, CEDiscriminator
    from trainer import train_CE
    from outpainting import *
    import time
    start_time = time.time()

    print("PyTorch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)

    # Define paths
    model_save_path = 'saved_models'
    html_save_path = 'outpaint_html'
    train_dir = "/mnt/Data/yangbo/data/Places365/train_large"
    val_dir = "/mnt/Data/yangbo/data/Places365/val_large"
    test_dir = "/mnt/Data/yangbo/data/Places365/test_large"

    # Define datasets & transforms
    batch_size = 16
    input_shape = (256,256)
    my_tf = transforms.Compose([
        transforms.CenterCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    train_data = CEImageDataset(train_dir, my_tf, input_shape)
    val_data   = CEImageDataset(val_dir, my_tf, input_shape)
    test_data  = CEImageDataset(test_dir, my_tf, input_shape)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    # Define model & device
    device = torch.device('cuda:0')
    device_ids = [0,1] # logical indeces of gpu
    # device = torch.device('cpu')
    G_net = CEGenerator()
    D_net = CEDiscriminator()
    G_net.apply(weights_init_normal)
    D_net.apply(weights_init_normal)
    G_net = nn.DataParallel(G_net, device_ids=device_ids)
    D_net = nn.DataParallel(D_net, device_ids=device_ids)
    G_net.to(device)
    D_net.to(device)
    print('device:', device)

    # Define losses
    criterion_pxl = nn.L1Loss()
    criterion_D = nn.MSELoss()
    optimizer_G = optim.Adam(G_net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D_net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    criterion_pxl.to(device)
    criterion_D.to(device)

    # Start training
    data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader} # NOTE: test is evidently not used by the train method
    n_epochs = 10
    adv_weight = [0.001, 0.005, 0.015, 0.040] # corresponds to epochs 1-10, 10-30, 30-60, 60-onwards
    hist_loss = train_CE(G_net, D_net, device, criterion_pxl, criterion_D, optimizer_G, optimizer_D,
                         data_loaders, model_save_path, html_save_path, n_epochs=n_epochs, adv_weight=adv_weight, input_shape=input_shape)


    
    
