#!/usr/bin/env python
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from common import *
import sys
import os
from trainers import *
import cv2
import torchvision
from tools import *
from optparse import OptionParser
from projects.pt_semantic_segmentation.lib.enet import ENet
from zoo.pytorch.utils import save_checkpoint, wrap_cuda, load_checkpoint
parser = OptionParser()
parser.add_option('--trans_alone', type=int, help="showing the translated image alone", default=0)
parser.add_option('--a2b', type=int, help="1 for a2b and others for b2a", default=1)
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--config',type=str,help="net configuration")
parser.add_option('--weights',type=str,help="file location to the trained generator network weights")
parser.add_option('--output_folder',type=str,help="output image folder")
parser.add_option('--save_segm', type=int, help="save segmentation of output image", default=0)

def init_segmentation():
  n_classes = 35
  segm_model = wrap_cuda(ENet(n_classes))
  #checkpoint = load_checkpoint('/staging/experiments/domain_adaptation/cityscapes_segmentation/20171202_202723/model_best.pth.tar') # 19 classes
  checkpoint = load_checkpoint('/staging/dadl/checkpoints/enet_pytorch/20180322_160509/checkpoint.pth.tar')
  segm_model.load_state_dict(checkpoint['state_dict'])
  segm_model.cuda()
  segm_model.eval()
  return segm_model

def main(argv):
  (opts, args) = parser.parse_args(argv)

  # Load experiment setting
  assert isinstance(opts, object)
  config = NetConfig(opts.config)

  ######################################################################################################################
  # Read training parameters from the yaml file
  hyperparameters = {}
  for key in config.hyperparameters:
    exec ('hyperparameters[\'%s\'] = config.hyperparameters[\'%s\']' % (key,key))

  if opts.a2b==1:
    dataset = config.datasets['train_a']
  else:
    dataset = config.datasets['train_b']
  exec ("data = %s(dataset)" % dataset['class_name'])
  root = dataset['root']
  folder = dataset['folder']
  list = dataset['list_name']
  list_fullpath = os.path.join(root, list)
  with open(list_fullpath) as f:
    content = f.readlines()
  image_list = [x.strip().split(' ')[0] for x in content]
  image_list.sort()

  cmd = "trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer']
  local_dict = locals()
  exec(cmd,globals(),local_dict)
  trainer = local_dict['trainer']

  # Prepare network
  trainer.gen.load_state_dict(torch.load(opts.weights))
  trainer.cuda(opts.gpu)
  # trainer.gen.eval()

  if opts.save_segm == 1:
    segm_model = init_segmentation()

  for image_name in image_list:
    print(image_name)
    full_img_name = os.path.join(root, folder, image_name)
    img = data._load_one_image(full_img_name,test=True)
    raw_data = img.transpose((2, 0, 1))  # convert to HWC
    final_data = torch.FloatTensor((raw_data / 255.0 - 0.5) * 2)
    final_data = final_data.contiguous()
    final_data = Variable(final_data.view(1,final_data.size(0),final_data.size(1),final_data.size(2))).cuda(opts.gpu)
    # trainer.gen.eval()
    if opts.a2b == 1:
      output_data = trainer.gen.forward_a2b(final_data)
    else:
      output_data = trainer.gen.forward_b2a(final_data)

    output_image_name = os.path.join(opts.output_folder, image_name)
    directory = os.path.dirname(output_image_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if opts.trans_alone == 0:
      if opts.save_segm == 1:
        segm_image__gt = segm_model.forward(final_data)[0]
        _, max_segm_image = torch.max(segm_image__gt, dim=1, keepdim=True)
        max_segm_image_gt = max_segm_image.expand(final_data.size()).float()
        max_segm_image_gt.data = (max_segm_image_gt.data / 35.0 - 0.5) *2 # 35 classes
        assembled_images = torch.cat((final_data, output_data[0], max_segm_image_gt), 3)
      else:
        assembled_images = torch.cat((final_data, output_data[0]), 3)
      torchvision.utils.save_image(assembled_images.data / 2.0 + 0.5, output_image_name)
    else:
      output_img = output_data[0].data.cpu().numpy()
      new_output_img = np.transpose(output_img, [2, 3, 1, 0])
      new_output_img = new_output_img[:, :, :, 0]
      out_img = np.uint8(255 * (new_output_img / 2.0 + 0.5))
      out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
      cv2.imwrite(output_image_name, out_img)
  return 0


if __name__ == '__main__':
  main(sys.argv)
