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
from glob import glob
import torchvision.transforms.functional as F

label_colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], 
                [0.929, 0.694, 0.125], [0.494, 0.184, 0.556]]

max_images = 100

opt = TestOptions().parse()

# dataloader = data.create_dataloader(opt)

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s' % (opt.subfolder))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Subfolder = %s' %
                    (opt.name, opt.subfolder))

# test
for img_dir in glob(os.path.join(web_dir, 'images/*'))[:max_images]:

    visualizer.save_webpage(webpage, img_dir)
#create js file named serve.js
with open(os.path.join(web_dir, 'serve.js'), 'w') as file:
    file.write("""
const express = require('express')
const app = express()
const path = require('path');

const port = 3000

app.use(express.static(__dirname + '/images'));

app.get('/', (req, res) => {
res.sendFile(path.join(__dirname, '/index.html'));
})

app.listen(port, () => {
console.log(`Server listening on port ${port}`)
})
""".strip())

webpage.save()
