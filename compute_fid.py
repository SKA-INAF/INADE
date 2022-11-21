from torchmetrics.image.fid import FrechetInceptionDistance as FID
import argparse
import glob
from PIL import Image
import torchvision.transforms.functional as F
from tqdm import tqdm 

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())
def main(args):
    #compute fid
    fid = FID().to(args.device)

    for image in tqdm(glob.glob(f'datasets/radiogalaxy/test/images/*.png'), desc='Iterating real images'):
        image = Image.open(image)
        image = image.convert('RGB')
        image = F.to_tensor(image) 
        im_tensor = (normalize(image) * 255).byte().unsqueeze(0).to(args.device)
        fid.update(im_tensor, real=True)

    for image in tqdm(glob.glob(f'results/{args.name}/{args.subfolder}/images/**/synthesized_image.png'), desc='Iterating generated images'):
        image = Image.open(image)
        image = image.convert('RGB')
        image = F.to_tensor(image) 
        im_tensor = (normalize(image) * 255).byte().unsqueeze(0).to(args.device)
        fid.update(im_tensor, real=False)

    # for image_dir in tqdm(glob.glob(f'results/{args.name}/test_latest/images/sample*')):
    #     orig_path = Path(image_dir) / 'original_image.png'
    #     gen_path = Path(image_dir) / 'synthesized_image.png'
    #     original = Image.open(orig_path)
    #     original = original.convert('RGB')
    #     original = F.to_tensor(original) 
    #     original_tensor = (normalize(original) * 255).byte().unsqueeze(0).to(args.device)
    #     fid.update(original_tensor, real=True)

    #     generated = Image.open(gen_path)
    #     generated = generated.convert('RGB')
    #     generated = F.to_tensor(generated) 
    #     generated_tensor = (normalize(generated) * 255).byte().unsqueeze(0).to(args.device)
    #     fid.update(generated_tensor, real=False)

    print(f'FID: {fid.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='rg-test', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--subfolder', type=str, default='test_latest', help='name of the experiment variation.')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    args = parser.parse_args()
    main(args)