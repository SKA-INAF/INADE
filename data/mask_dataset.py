from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from pathlib import Path
from data.base_dataset import get_params, get_transform
from PIL import Image
import torch

class MaskDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=4)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        # parser.set_defaults(dataroot='/home/tzt/HairSynthesis/SPADE/datasets/ADEChallengeData2016/')
        return parser

    def get_paths(self, opt):
        root = Path(opt.dataroot)
        # phase = 'val' if opt.phase == 'test' else 'train'

        all_images = make_dataset(root / opt.phase, recursive=True, read_cache=False, write_cache=False)
        image_paths = []
        label_paths = []
        instance_paths = []
        sketch_paths = []
        for p in all_images:
            # if '_%s_' % phase not in p:
            #     continue
            if p.endswith('.png') and 'images' in p:
                image_paths.append(p)
            elif 'mask' in p and p.endswith('.png'):
                label_paths.append(p)
            elif 'instances' in p and p.endswith('.png'):
                instance_paths.append(p)
            elif 'edgesD' in p and p.endswith('.png') and opt.add_sketch:
                sketch_paths.append(p)

        return label_paths, image_paths, instance_paths, sketch_paths

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        # label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        label_tensor = torch.where(label_tensor != 0, self.opt.label_nc, 0.)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        # if using sketch maps
        if not self.opt.add_sketch:
            sketch_tensor = 0
        else:
            # sketch range is 0 and 255
            sketch_path = self.sketch_paths[index]
            sketch = Image.open(sketch_path)
            sketch_tensor = transform_label(sketch)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': torch.rand(3, *label_tensor.shape[1:]), # HACK to avoid breaking everything, I don't need the image
                      'sketch': sketch_tensor,
                      'path': label_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
    
    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc
