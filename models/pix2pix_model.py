import torch
import models.networks as networks
import util.util as util
import random
try:
    from torch.cuda.amp import autocast as autocast, GradScaler
    AMP = True
except:
    AMP = False

class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        torch_version = torch.__version__.split('.')
        if int(torch_version[1]) >= 2:
            self.ByteTensor = torch.cuda.BoolTensor if self.use_gpu() \
                else torch.BoolTensor
        else:
            self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
                else torch.ByteTensor

        self.netG, self.netD, self.netE, self.netIE = self.initialize_networks(opt)

        self.isNoise = True if 'inade' in opt.norm_mode else False

        self.amp = True if AMP and opt.use_amp and opt.isTrain else False

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids, self.isNoise)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, noise=None, noise_ins=None):
        input_semantics, real_image, input_instances, sketch = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, input_instances, sketch)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, input_instances, sketch)
            return d_loss
        elif mode == 'encode_only' and 'spade' in self.opt.norm_mode:
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image, input_instances, sketch)
            return fake_image
        elif mode == 'demo':
            with torch.no_grad():
                fake_image = self.demo_generate_fake(input_semantics, real_image, input_instances, sketch, noise, noise_ins, data)
                return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            if 'spade' in opt.norm_mode:
                G_params += list(self.netE.parameters())
            elif 'inade' in opt.norm_mode:
                G_params += list(self.netIE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            if 'spade' in self.opt.norm_mode:
                util.save_network(self.netE, 'E', epoch, self.opt)
            elif 'inade' in self.opt.norm_mode:
                util.save_network(self.netIE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae and 'spade' in opt.norm_mode else None
        netIE = networks.define_IE(opt) if opt.use_vae and 'inade' in opt.norm_mode else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                if 'spade' in opt.norm_mode:
                    netE = util.load_network(netE, 'E', opt.which_epoch, opt)
                elif 'inade' in opt.norm_mode:
                    netIE = util.load_network(netIE, 'E', opt.which_epoch, opt)

        return netG, netD, netE, netIE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()
            data['sketch'] = data['sketch'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        # create one-hot instance map
        if 'inade' in self.opt.norm_mode:
            inst_map = data['instance'].long()
            bs, _, h, w = inst_map.size()
            nc = inst_map.max()+1
            input_inst = self.FloatTensor(bs, nc, h, w).zero_()
            input_instances = input_inst.scatter_(1, inst_map, 1.0)
        else:
            input_instances = None

        return input_semantics, data['image'], input_instances, data['sketch']

    def compute_generator_loss(self, input_semantics, real_image, input_instances, sketch):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, input_instances, sketch, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    if not self.isNoise:
                        GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                    else:
                        if j >=3:
                            GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            if self.amp:
                with autocast():
                    G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
            else:
                G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, input_instances, sketch):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image, input_instances, sketch)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        if self.amp:
            with autocast():
                mu, logvar = self.netE(real_image)
        else:
            mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def instance_encode_z(self, real_image, input_instances):
        if self.amp:
            with autocast():
                s_mus, s_logvars, b_mus, b_logvars = self.netIE(real_image,input_instances)
        else:
            s_mus, s_logvars, b_mus, b_logvars = self.netIE(real_image,input_instances)
        z = [s_mus,torch.exp(0.5 * s_logvars),b_mus,torch.exp(0.5 * b_logvars)]
        return z, s_mus, s_logvars, b_mus, b_logvars

    def generate_fake(self, input_semantics, real_image, input_instances, sketch, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            if 'spade' in self.opt.norm_mode:
                z, mu, logvar = self.encode_z(real_image)
                if compute_kld_loss:
                    KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
            elif 'inade' in self.opt.norm_mode:
                z, s_mus, s_logvars, b_mus, b_logvars = self.instance_encode_z(real_image,input_instances)
                if compute_kld_loss:
                    KLD_loss = (self.KLDLoss(s_mus, s_logvars)+self.KLDLoss(b_mus, b_logvars)) * self.opt.lambda_kld / 2

        if self.amp:
            with autocast():
                fake_image = self.netG(input_semantics, z=z, input_instances=input_instances, sketch=sketch)
        else:
            fake_image = self.netG(input_semantics, z=z, input_instances=input_instances, sketch=sketch)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        if self.amp:
            with autocast():
                discriminator_out = self.netD(fake_and_real)
        else:
            discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def demo_generate_fake(self, input_semantics, real_image, input_instances, sketch, noise, noise_ins, data):
        # process ref inputs
        data['ref_label'] = data['ref_label'].long()
        if self.use_gpu():
            data['ref_label'] = data['ref_label'].cuda()
            data['ref_instance'] = data['ref_instance'].cuda()
            data['ref_image'] = data['ref_image'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot instance map
        ref_inst_map = data['ref_instance'].long()
        bs, _, h, w = ref_inst_map.size()
        nc = ref_inst_map.max()+1
        ref_input_inst = self.FloatTensor(bs, nc, h, w).zero_()
        ref_input_instances = ref_input_inst.scatter_(1, ref_inst_map, 1.0)

        # reference encoder
        z, _, _, _, _ = self.instance_encode_z(data['ref_image'], ref_input_instances)

        # init the z with zero mu and one std
        inst_nc = input_instances.shape[1]
        s_mus = torch.zeros([1, inst_nc, self.opt.noise_nc]).cuda()
        s_stds = torch.ones([1, inst_nc, self.opt.noise_nc]).cuda()
        z_0 = [s_mus, s_stds, s_mus, s_stds]

        # change the z_0 from z if necessary
        if data['is_ref_inst'] and data['ref_inst_id'] != -1:
            for idx in range(len(z_0)):
                z_0[idx][:,data['inst_id'],:] = z[idx][:,data['ref_inst_id'],:]

        fake_image = self.netG(input_semantics, z=z_0, input_instances=input_instances, sketch=sketch, noise=noise, noise_ins=noise_ins)

        return fake_image
