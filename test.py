"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from PIL import Image
import torchvision.transforms.functional as F

label_colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], 
                [0.929, 0.694, 0.125], [0.494, 0.184, 0.556]]

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s' % (opt.subfolder))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Subfolder = %s' %
                    (opt.name, opt.subfolder))


# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        image = Image.open(img_path[b])
        im_tensor = F.to_tensor(image) * 2 - 1
        label_path = img_path[b].replace('images', 'masks')
        label = Image.open(label_path) 
        lbl_tensor = F.to_tensor(label) * 2 - 1
        visuals = OrderedDict([('original_image', im_tensor),
                               ('input_label', lbl_tensor),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
