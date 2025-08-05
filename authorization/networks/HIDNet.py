
import torch
import torch.nn as nn
import numpy as np
from networks.mapper import MappingNetwork
from networks.encoder_decoder import *
from networks.discriminator import Discriminator
from networks.utils import VGGLoss
import yaml
import os

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

class HIDNet:

    def __init__(self,device,gamma=0.5,input_mask=None,randomFlag=False,pixel_space=False) -> None:
        super().__init__()
        self.device = device
        self.mapping_network = nn.Identity()
        config_path = os.path.abspath(f'../configs/authorization.yaml')
        self.config = load_config(config_path)
        self.encoder_decoder = EncoderDecoder(encode_message_length=self.config['message_length'],decode_message_length=self.config['message_length'],pixel_space=pixel_space,block_size=self.config['block_size'],).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.randomFlag = randomFlag
        self.pixel_space = pixel_space
        if not randomFlag:
            self.random_mask = torch.bernoulli(torch.ones(3,512,512) * gamma).to(self.device)
            self.gamma = gamma
            if input_mask is not None:
                self.random_mask = torch.load(input_mask).to(self.device)
            # print(self.random_mask.sum()/self.random_mask.reshape(-1).shape[0])
        else:
            self.gamma = gamma
        self.init_optimizer()
        self.init_losses()
        


    def init_optimizer(self):
        self.optimizer_enc_dec = torch.optim.Adam(list(self.encoder_decoder.parameters()) + list(self.mapping_network.parameters()),lr=self.config['train']['lr'])
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters(),lr=self.config['train']['lr'])

    def init_losses(self):
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.vgg_loss = None 
        self.adversarial_loss  = self.config['train']['adversarial_loss']
        self.encoder_loss = self.config['train']['encoder_loss']
        self.decoder_loss = self.config['train']['decoder_loss']
        self.reg_loss = self.config['train']['reg_loss']

    def train_one_batch(self,batch):

        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), 1.0, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), 0.0, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), 1.0, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()

            # train on fake
            if not self.randomFlag:
                encoded_images, decoded_messages, random_mask,encoded_dct,img_dct_masked = self.encoder_decoder(images,self.mapping_network(messages),random_mask=self.random_mask)
            else:
                random_mask = torch.bernoulli(torch.ones(images.shape[0],3,512,512) * self.gamma).to(self.device)
                encoded_images, decoded_messages, random_mask,encoded_dct,img_dct_masked = self.encoder_decoder(images,self.mapping_network(messages),random_mask=random_mask)
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc) + self.mse_loss(encoded_images, images)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss_reg = self.mse_loss(encoded_dct, img_dct_masked)
            if not self.pixel_space:
                reg_co = self.reg_loss
            else:
                reg_co = 0
            g_loss = self.adversarial_loss * g_loss_adv + self.encoder_loss * g_loss_enc \
                     + self.decoder_loss * (g_loss_dec+reg_co*g_loss_reg)

            g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'dec_reg        ': g_loss_reg.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, random_mask, decoded_messages)

    def val_one_batch(self,batch):

        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
 
            d_target_label_cover = torch.full((batch_size, 1), 1.0, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), 0.0, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), 1.0, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
       
            if not self.randomFlag:
                encoded_images, decoded_messages, random_mask,encoded_dct,img_dct_masked = self.encoder_decoder(images,self.mapping_network(messages),random_mask=self.random_mask)
            else:
                random_mask = torch.bernoulli(torch.ones(images.shape[0],3,512,512) * self.gamma).to(self.device)
                encoded_images, decoded_messages, random_mask,encoded_dct,img_dct_masked = self.encoder_decoder(images,self.mapping_network(messages),random_mask=random_mask)
            
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss_reg = self.mse_loss(encoded_dct, img_dct_masked)
            if not self.pixel_space:
                reg_co = self.reg_loss
            else:
                reg_co = 0
            g_loss = self.adversarial_loss * g_loss_adv + self.encoder_loss * g_loss_enc \
                     + self.decoder_loss * (g_loss_dec+reg_co*g_loss_reg)

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
 
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'dec_reg        ': g_loss_reg.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, random_mask, decoded_messages)



        


