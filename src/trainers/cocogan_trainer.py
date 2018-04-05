"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from .cocogan_nets import *
from .helpers import get_model_list, _compute_fake_acc, _compute_true_acc
from .init import *
import torch
import torch.nn as nn
import os
import itertools


class COCOGANTrainer(nn.Module):
  def __init__(self, hyperparameters):
    super(COCOGANTrainer, self).__init__()
    lr = hyperparameters['lr']
    # Initiate the networks
    exec( 'self.dis = %s(hyperparameters[\'dis\'])' % hyperparameters['dis']['name'])
    exec( 'self.gen = %s(hyperparameters[\'gen\'])' % hyperparameters['gen']['name'] )
    # Setup the optimizers
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    # Network weight initialization
    self.dis.apply(gaussian_weights_init)
    self.gen.apply(gaussian_weights_init)
    # Setup the loss function for training
    self.ll_loss_criterion_a = torch.nn.L1Loss()
    self.ll_loss_criterion_b = torch.nn.L1Loss()
    self.ll_loss_criterion_c = torch.nn.L1Loss()


  def _compute_kl(self, mu):
    # def _compute_kl(self, mu, sd):
    # mu_2 = torch.pow(mu, 2)
    # sd_2 = torch.pow(sd, 2)
    # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
    # return encoding_loss
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def gen_update(self, images_a, images_b, images_c, hyperparameters):
    self.gen.zero_grad()
    x_aa, x_ba, x_ca, x_ab, x_bb, x_cb, x_ac, x_bc, x_cc, shared = self.gen(images_a, images_b, images_c)
    x_bab, shared_bab = self.gen.forward_a2b(x_ba)
    x_aba, shared_aba = self.gen.forward_b2a(x_ab)
    x_bcb, shared_bcb = self.gen.forward_c2b(x_bc)
    x_cbc, shared_cbc = self.gen.forward_b2c(x_cb)
    x_cac, shared_cac = self.gen.forward_a2c(x_ca)
    x_aca, shared_aca = self.gen.forward_c2a(x_ac)
    outs_ba, outs_ab, outs_ca, outs_cb, outs_ac, outs_bc = self.dis(x_ba, x_ab, x_ca, x_cb, x_ac, x_bc)
    for it, (out_ba, out_ab, out_ca, out_cb, out_ac, out_bc) in enumerate(itertools.izip(outs_ba, outs_ab, outs_ca, outs_cb, outs_ac, outs_bc)):
      outputs_ba = nn.functional.sigmoid(out_ba)
      outputs_ab = nn.functional.sigmoid(out_ab)
      outputs_ca = nn.functional.sigmoid(out_ca)
      outputs_cb = nn.functional.sigmoid(out_cb)
      outputs_ac = nn.functional.sigmoid(out_ac)
      outputs_bc = nn.functional.sigmoid(out_bc)

      all_ones = Variable(torch.ones((outputs_ba.size(0))).cuda(self.gpu))
      if it==0:
        ad_loss_ba = nn.functional.binary_cross_entropy(outputs_ba, all_ones)
        ad_loss_ab = nn.functional.binary_cross_entropy(outputs_ab, all_ones)
        ad_loss_ca = nn.functional.binary_cross_entropy(outputs_ca, all_ones)
        ad_loss_cb = nn.functional.binary_cross_entropy(outputs_cb, all_ones)
        ad_loss_ac = nn.functional.binary_cross_entropy(outputs_ac, all_ones)
        ad_loss_bc = nn.functional.binary_cross_entropy(outputs_bc, all_ones)
      else:
        ad_loss_ba += nn.functional.binary_cross_entropy(outputs_ba, all_ones)
        ad_loss_ab += nn.functional.binary_cross_entropy(outputs_ab, all_ones)
        ad_loss_ca += nn.functional.binary_cross_entropy(outputs_ca, all_ones)
        ad_loss_cb += nn.functional.binary_cross_entropy(outputs_cb, all_ones)
        ad_loss_ac += nn.functional.binary_cross_entropy(outputs_ac, all_ones)
        ad_loss_bc += nn.functional.binary_cross_entropy(outputs_bc, all_ones)

    enc_loss  = self._compute_kl(shared)
    enc_bab_loss = self._compute_kl(shared_bab)
    enc_aba_loss = self._compute_kl(shared_aba)
    enc_bcb_loss = self._compute_kl(shared_bcb)
    enc_cbc_loss = self._compute_kl(shared_cbc)
    enc_cac_loss = self._compute_kl(shared_cac)
    enc_aca_loss = self._compute_kl(shared_aca)

    ll_loss_a = self.ll_loss_criterion_a(x_aa, images_a)
    ll_loss_b = self.ll_loss_criterion_b(x_bb, images_b)
    ll_loss_c = self.ll_loss_criterion_c(x_cc, images_c)

    ll_loss_aba = self.ll_loss_criterion_a(x_aba, images_a)
    ll_loss_bab = self.ll_loss_criterion_b(x_bab, images_b)
    ll_loss_bcb = self.ll_loss_criterion_a(x_bcb, images_b)
    ll_loss_cbc = self.ll_loss_criterion_b(x_cbc, images_c)
    ll_loss_cac = self.ll_loss_criterion_a(x_cac, images_c)
    ll_loss_aca = self.ll_loss_criterion_b(x_aca, images_a)

    total_loss = hyperparameters['gan_w'] * (ad_loss_ba + ad_loss_ab + ad_loss_ca + ad_loss_cb + ad_loss_ac + ad_loss_bc) + \
                 hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b + ll_loss_c) + \
                 hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab + ll_loss_bcb + ll_loss_cbc + ll_loss_cac + ll_loss_aca) + \
                 hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss) + \
                 hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss + enc_bcb_loss + enc_cbc_loss + enc_cac_loss + enc_aca_loss)
    total_loss.backward()
    self.gen_opt.step()
    self.gen_enc_loss = enc_loss.data.cpu().numpy()[0] #TODO: Eventually add new loss terms to here as well if I figure out why they need to be numpyed
    self.gen_enc_bab_loss = enc_bab_loss.data.cpu().numpy()[0]
    self.gen_enc_aba_loss = enc_aba_loss.data.cpu().numpy()[0]
    self.gen_ad_loss_ba = ad_loss_ba.data.cpu().numpy()[0]
    self.gen_ad_loss_ab = ad_loss_ab.data.cpu().numpy()[0]
    self.gen_ll_loss_a = ll_loss_a.data.cpu().numpy()[0]
    self.gen_ll_loss_b = ll_loss_b.data.cpu().numpy()[0]
    self.gen_ll_loss_aba = ll_loss_aba.data.cpu().numpy()[0]
    self.gen_ll_loss_bab = ll_loss_bab.data.cpu().numpy()[0]
    self.gen_total_loss = total_loss.data.cpu().numpy()[0]
    return (x_aa, x_ba, x_ab, x_bb, x_aba, x_bab)

  def dis_update(self, images_a, images_b, images_c,hyperparameters):
    self.dis.zero_grad()
    x_aa, x_ba, x_ca, x_ab, x_bb, x_cb, x_ac, x_bc, x_cc, shared = self.gen(images_a, images_b, images_c)
    data_ba = torch.cat((images_a, x_ba), 0)
    data_ab = torch.cat((images_b, x_ab), 0)
    data_ca = torch.cat((images_a, x_ca), 0)
    data_cb = torch.cat((images_b, x_cb), 0)
    data_ac = torch.cat((images_c, x_ac), 0)
    data_bc = torch.cat((images_c, x_bc), 0)
    outs_ba, outs_ab, outs_ca, outs_cb, outs_ac, outs_bc = self.dis(data_ba, data_ab, data_ca, data_cb, data_ac, data_bc)
    # res_true_a, res_true_b = self.dis(images_a,images_b)
    # res_fake_a, res_fake_b = self.dis(x_ba, x_ab)
    for it, (out_ba, out_ab, out_ca, out_cb, out_ac, out_bc) in enumerate(itertools.izip(outs_ba, outs_ab, outs_ca, outs_cb, outs_ac, outs_bc)):
      out_ba = nn.functional.sigmoid(out_ba)
      out_ab = nn.functional.sigmoid(out_ab)
      out_ca = nn.functional.sigmoid(out_ca)
      out_cb = nn.functional.sigmoid(out_cb)
      out_ac = nn.functional.sigmoid(out_ac)
      out_bc = nn.functional.sigmoid(out_bc)
      out_true_ba, out_fake_ba = torch.split(out_ba, out_ba.size(0) // 2, 0)
      out_true_ab, out_fake_ab = torch.split(out_ab, out_ab.size(0) // 2, 0)
      out_true_ca, out_fake_ca = torch.split(out_ca, out_ca.size(0) // 2, 0)
      out_true_cb, out_fake_cb = torch.split(out_cb, out_cb.size(0) // 2, 0)
      out_true_ac, out_fake_ac = torch.split(out_ba, out_ac.size(0) // 2, 0)
      out_true_bc, out_fake_bc = torch.split(out_ab, out_bc.size(0) // 2, 0)
      out_true_n = out_true_ba.size(0)
      out_fake_n = out_fake_ba.size(0)
      all1 = Variable(torch.ones((out_true_n)).cuda(self.gpu))
      all0 = Variable(torch.zeros((out_fake_n)).cuda(self.gpu))
      ad_true_loss_ba = nn.functional.binary_cross_entropy(out_true_ba, all1)
      ad_true_loss_ab = nn.functional.binary_cross_entropy(out_true_ab, all1)
      ad_fake_loss_ba = nn.functional.binary_cross_entropy(out_fake_ba, all0)
      ad_fake_loss_ab = nn.functional.binary_cross_entropy(out_fake_ab, all0)
      ad_true_loss_ca = nn.functional.binary_cross_entropy(out_true_ca, all1)
      ad_true_loss_ac = nn.functional.binary_cross_entropy(out_true_ac, all1)
      ad_fake_loss_ca = nn.functional.binary_cross_entropy(out_fake_ca, all0)
      ad_fake_loss_ac = nn.functional.binary_cross_entropy(out_fake_ac, all0)
      ad_true_loss_cb = nn.functional.binary_cross_entropy(out_true_cb, all1)
      ad_true_loss_bc = nn.functional.binary_cross_entropy(out_true_bc, all1)
      ad_fake_loss_cb = nn.functional.binary_cross_entropy(out_fake_cb, all0)
      ad_fake_loss_bc = nn.functional.binary_cross_entropy(out_fake_bc, all0)
      if it==0:
        ad_loss_ba = ad_true_loss_ba + ad_fake_loss_ba
        ad_loss_ab = ad_true_loss_ab + ad_fake_loss_ab
        ad_loss_ca = ad_true_loss_ca + ad_fake_loss_ca
        ad_loss_ac = ad_true_loss_ac + ad_fake_loss_ac
        ad_loss_bc = ad_true_loss_bc + ad_fake_loss_bc
        ad_loss_cb = ad_true_loss_cb + ad_fake_loss_cb
      else:
        ad_loss_ba += ad_true_loss_ba + ad_fake_loss_ba
        ad_loss_ab += ad_true_loss_ab + ad_fake_loss_ab
        ad_loss_ca += ad_true_loss_ca + ad_fake_loss_ca
        ad_loss_ac += ad_true_loss_ac + ad_fake_loss_ac
        ad_loss_bc += ad_true_loss_bc + ad_fake_loss_bc
        ad_loss_cb += ad_true_loss_cb + ad_fake_loss_cb

      true_a_acc = _compute_true_acc(out_true_ba)  #Why?
      true_b_acc = _compute_true_acc(out_true_ab)
      fake_a_acc = _compute_fake_acc(out_fake_ba)
      fake_b_acc = _compute_fake_acc(out_fake_ab)
      exec( 'self.dis_true_acc_%d = 0.5 * (true_a_acc + true_b_acc)' %it) #Why?
      exec( 'self.dis_fake_acc_%d = 0.5 * (fake_a_acc + fake_b_acc)' %it)
    loss = hyperparameters['gan_w'] * ( ad_loss_ba + ad_loss_ab + ad_loss_ca + ad_loss_ac + ad_loss_bc + ad_loss_cb)
    loss.backward()
    self.dis_opt.step()
    self.dis_loss = loss.data.cpu().numpy()[0]
    return

  def assemble_outputs(self, images_a, images_b, network_outputs):
    images_a = self.normalize_image(images_a)
    images_b = self.normalize_image(images_b)
    x_aa = self.normalize_image(network_outputs[0])
    x_ba = self.normalize_image(network_outputs[1])
    x_ab = self.normalize_image(network_outputs[2])
    x_bb = self.normalize_image(network_outputs[3])
    x_aba = self.normalize_image(network_outputs[4])
    x_bab = self.normalize_image(network_outputs[5])
    return torch.cat((images_a[0:1, ::], x_aa[0:1, ::], x_ab[0:1, ::], x_aba[0:1, ::],
                      images_b[0:1, ::], x_bb[0:1, ::], x_ba[0:1, ::], x_bab[0:1, ::]), 3)

  def resume(self, snapshot_prefix):
    dirname = os.path.dirname(snapshot_prefix)
    last_model_name = get_model_list(dirname,"gen")
    if last_model_name is None:
      return 0
    self.gen.load_state_dict(torch.load(last_model_name))
    iterations = int(last_model_name[-12:-4])
    last_model_name = get_model_list(dirname, "dis")
    self.dis.load_state_dict(torch.load(last_model_name))
    print('Resume from iteration %d' % iterations)
    return iterations

  def save(self, snapshot_prefix, iterations):
    gen_filename = '%s_gen_%08d.pkl' % (snapshot_prefix, iterations + 1)
    dis_filename = '%s_dis_%08d.pkl' % (snapshot_prefix, iterations + 1)
    torch.save(self.gen.state_dict(), gen_filename)
    torch.save(self.dis.state_dict(), dis_filename)

  def cuda(self, gpu):
    self.gpu = gpu
    self.dis.cuda(gpu)
    self.gen.cuda(gpu)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
