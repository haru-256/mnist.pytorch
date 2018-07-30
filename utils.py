from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy

# custom weights initialization


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # Conv系全てに対しての初期化
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # initialize for BN
        m.bias.data.fill_(0)


def standard_gan_train(models, datasets, optimizers, num_epochs=30, batch_size=128,
                       device, scheduler=None):
    since = datetime.datetime.now()
    epochs = tqdm(range(num_epochs), desc="Epoch", unit='epoch')
    # construct dataloader
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size,
                                                     shuffle=True, num_workers=2)
    dataset_sizes = len(dataloader)
    criterion = nn.CrossEntropyLoss()
    # train loop
    for epoch in epochs:
            iteration = tqdm(dataloader,
                             desc="Iteration",
                             unit='iter')
            epoch_dis_loss = 0.0
            epoch_gen_loss = 0.0
            for inputs, _ in iteration:
                inputs = inputs.to(device)
                real_labels = torch.ones(batch_size, device=device)
                fake_labels = torch.zeros(batch_size, device=device)
                lables = {'real': real_labels, 'fake': fake_labels}
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                dis_loss = train_dis(models, optimizers,
                 inputs, labels, criterion)
                ############################
                # (2) Update G netwrk: maximize log(D(G(z)))
                ###########################
                fake_labels = torch.ones_like(fake_labels, device=device)
                gen_loss = train_gen(
                    models['generator'], optimizers['generator'],
                     fake_labels, criterion)
                # statistics
                epoch_dis_loss += dis_loss.item() * inputs.size(0)
                epoch_gen_loss += gen_loss.item() * inputs.size(0)

            # print loss
            epoch_dis_loss /= dataset_sizes
            epoch_gen_loss /= dataset_sizes
            tqdm.write('Epoch: {} GenLoss: {:.4f} DisLoss: {:.4f}'.format(
                epoch, epoch_gen_loss, epoch_dis_loss))

        tqdm.write("")

    time_elapsed = datetime.datetime.now() - since
    tqdm.write('Training complete in {}'.format(time_elapsed))

    return model

def train_dis(models, optimizers, data, lables, criterion):
    gen, dis = models['generator'], models['discriminator']
    gen_optim, dis_optim = optimizers['generator'], optimizers['discriminator']

    # train disctiminator
    dis.train()
    # zero the parameter gradientsreal
    dis_optim.zero_grad()

    # make noise & forward 
    z = gen.make_hidden(data.size()[0])
    x_fake = gen(z)
    y_fake = dis(image_fake)
    y_real=dis(data)
    # calc loss for discriminator
    dis_loss = criterion(y_real, labels['real']) + criterion(y_fake, labels['fake'])
    
    # update parameters of discrimminator
    dis_loss.backward()
    dis_optim.step()

    return dis_loss

def train_gen(gen, gen_optim, fake_labels, criterion):
    # train generator
    gen.train()
    # zero the parameter gradientsreal
    gen_optim.zero_grad()
    # make noise & forward 
    z = gen.make_hidden(fake_labels.size()[0])
    x_fake = gen(z)
    y_fake = dis(image_fake)

    # calc loss for generator
    gen_loss = criterion(y_fake, fake_labels)

    # update parameters of discrimminator
    gen_loss.backward()
    gen_optim.step()

    return gen_loss 

