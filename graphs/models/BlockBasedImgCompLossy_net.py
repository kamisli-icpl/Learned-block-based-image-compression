import torch
from torch import nn
from torch.nn import functional as F
# from visdom import Visdom
from graphs.layers.masked_conv2d import MaskedConv2d, MaskedConv2d_zhat_x
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.layers import GDN, GDN1
import math
from compressai.ans import BufferedRansEncoder, RansDecoder


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(mmin=SCALES_MIN, mmax=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(mmin), math.log(mmax), levels))


class BlockBasedImgCompLossyNetv4(nn.Module):
    """ Neural-network architecture that implements Masked Convolution based Block-based Lossy Image Compression """
    def __init__(self, config):
        super(BlockBasedImgCompLossyNetv4, self).__init__()
        # set some parameters of the networks used in model
        B = config.block_size
        B2 = B**2
        K1, K2, K3, K4 = config.KS[0], config.KS[1], config.KS[2], config.KS[3]  # ks for each layer
        P1, P2, P3, P4 = K1//2, K2//2, K3//2, K4//2                  # padding for each layer
        C1 = config.N  # num channels for each layer
        C2 = config.N
        C3 = config.N
        C4 = config.M
        # PREDICTION and TRANSFORM networks
        # processes x + zhat to generate y, the latent variables to send to decoder
        self.prtr_forward1 = MaskedConv2d('B', B2 * 3, C1,  1, 1, 0)  # processes x, the original block
        self.prtr_forward2 = MaskedConv2d('A', B2 * 3, C1, K1, 1, P1)  # processes zhat, neighboring blocks
        self.prtr_forward3 = nn.Sequential(                          # processes cat of previous two outputs
            GDN(C1),
            MaskedConv2d('B',     C1, C2, K2, 1, P2),
            GDN(C2),
            MaskedConv2d('B',     C2, C3, K3, 1, P3),
            GDN(C3),
            MaskedConv2d('B',     C3, C4, K4, 1, P4)
        )
        # processes zhat + y_qnt to gnerate x_rec
        self.prtr_inverse1 = MaskedConv2d('B', C4,     C1,  1, 1, 0)  # processes y_qnt, the received latent variabls of blk
        self.prtr_inverse2 = MaskedConv2d('A', B2 * 3, C1, K1, 1, P1)  # processes zhat, neighboring blocks
        self.prtr_inverse3 = nn.Sequential(                           # processes cat of previous two outputs
            GDN(C1, inverse=True),
            MaskedConv2d('B',     C1, C2, K2, 1, P2),
            GDN(C2, inverse=True),
            MaskedConv2d('B',     C2, C3, K3, 1, P3),
            GDN(C3, inverse=True),
            MaskedConv2d('B',     C3, B2*3, K4, 1, P4)
        )
        # NEIGHBORBLOCKS CONDITIONING network (generates mean+scale for conditional gaussian)
        C1 = C1  # num channels for each layer
        C2 = C2
        C3 = C3
        C4 = C4 * 2  # need channels for mean and scale
        # processes zhat to generate means, stdevs
        self.get_meanscale = nn.Sequential(
            MaskedConv2d('A', B2 * 3, C1, K1, 1, P1, bias=True, padding_mode='zeros'),  # use A :zhat dreictly input
            nn.LeakyReLU(),
            MaskedConv2d('B', C1, C2, K2, 1, P2),
            nn.LeakyReLU(),
            MaskedConv2d('B', C2, C3, K3, 1, P3),
            nn.LeakyReLU(),
            MaskedConv2d('B', C3, C4, K4, 1, P4)  # just use 1x1 conv in last layer. why ?
        )
        # CDF
        self.conditional_gaussian_model = GaussianConditional(None)
        # self.viz = Visdom()
        # put parts of the model into module list, which will be used to freeze updates in that part of the model
        # self.ae_enc = nn.ModuleList([self.prtr_forward1, self.prtr_forward2, self.prtr_forward3])
        # self.ae_dec = nn.ModuleList([self.prtr_inverse1, self.prtr_inverse2, self.prtr_inverse3])
        # self.en_mdl = nn.ModuleList([self.get_meanscale, self.conditional_gaussian_model])

    def forward_prtr_(self, zhat, x):
        out_x = self.prtr_forward1(x)
        out_zhat = self.prtr_forward2(zhat)
        return self.prtr_forward3(out_x + out_zhat)

    def inverse_prtr_(self, zhat, y_qnt):
        out_y_qnt = self.prtr_inverse1(y_qnt)
        out_zhat = self.prtr_inverse2(zhat)
        return self.prtr_inverse3(out_y_qnt + out_zhat)

    def forward(self, zhat, x):
        """
        See notes if can not easily understand architecture
        :param zhat: reconstructed images/patches  # B x C x H x W
        :param x: original images/patches  # B x C x H x W
        :return: reconstructed image, -log2(quantized latent tensor probability)  # B x C x H x W , B x Cy x Hy x Wy
        """
        # forward prediction + transform
        y = self.forward_prtr_(zhat, x)
        # get mean&scale for y
        ksi = self.get_meanscale(zhat)
        scales, means = ksi.chunk(2, dim=1)
        y_qnt, pmf_values_y = self.conditional_gaussian_model(y, scales, means=means)
        self_informations_y = -torch.log2(pmf_values_y)
        # inverse prediction + transform
        xhat = self.inverse_prtr_(zhat, y_qnt)
        return xhat, self_informations_y

    def freeze_some_network(self, part, freeze):
        if part == "aenc":
            nn_modu_list = nn.ModuleList([self.prtr_forward1, self.prtr_forward2, self.prtr_forward3])
        elif part == "adec":
            nn_modu_list = nn.ModuleList([self.prtr_inverse1, self.prtr_inverse2, self.prtr_inverse3])
        elif part == "emdl":
            nn_modu_list = nn.ModuleList([self.get_meanscale, self.conditional_gaussian_model])
        else:
            print('Warning! Incorrect network part name. Will not freeze network parameters i.e. requires_grad !!!')
        for i in range(0, len(nn_modu_list)):
            for name, p in nn_modu_list[i].named_parameters():
                p.requires_grad = not freeze

    def update(self, force=False):
        scale_table = get_scale_table()
        updated = self.conditional_gaussian_model.update_scale_table(scale_table, force=force)
        updated |= self.update_bottlenecks(force=force)
        return updated

    def update_bottlenecks(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.
        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.
        Args:
            force (bool): overwrite previous values (default: False)
        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.
        """
        updated = False
        for m in self.children():  # ?? m in self.modules():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def compress(self, x, LRU, chlat,):
        """ Compress one image, that has been reshaped to  1 x 3*B*B x H/B x W/B,  with arithmetic coding. """
        # prep stuff to pass to arithmetic coder
        cdf = self.conditional_gaussian_model.quantized_cdf.tolist()
        cdf_lengths = self.conditional_gaussian_model.cdf_length.tolist()
        offsets = self.conditional_gaussian_model.offset.tolist()
        self.conditional_gaussian_model._check_cdf_size()
        self.conditional_gaussian_model._check_cdf_length()
        self.conditional_gaussian_model._check_offsets_size()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        # get receptive field size
        L, R, U = LRU[0], LRU[1], LRU[2]
        # get inputs shape
        bt, ch, hg, wd = x.shape
        # tensors to convert to list later
        #symbols_list_tobe = torch.zeros((hg*wd, chlat), device=x.device, dtype=torch.int32)
        #indexes_list_tobe = torch.zeros((hg*wd, chlat), device=x.device, dtype=torch.int32)
        # get a reconstructed version of data which is initially all zeros
        zhat = torch.zeros_like(x)
        y_qnt = torch.zeros((bt, chlat, hg, wd), device=x.device)
        # run through model, arithmetic code, update one reconstruction pixel in zhat, do again...
        for v in range(0, hg):
            UU = max(0, v - U)
            for h in range(0, wd):
                LL = max(0, h - L)
                RR = min(wd, h + (R+1))
                indexes_blk, y_sym_blk, xhat_blk, y_qnt_blk = self.compress_blk(zhat[:, :, UU:v+1, LL:RR], x[:, :, UU:v+1, LL:RR], y_qnt[:, :, UU:v+1, LL:RR], v, h, LL, RR, UU)
                symbols_list.extend(y_sym_blk.squeeze().tolist())
                indexes_list.extend(indexes_blk.squeeze().tolist())
                #symbols_list_tobe[v*wd + h, :] = y_sym_blk.squeeze()
                #indexes_list_tobe[v*wd + h, :] = indexes_blk.squeeze()
                y_qnt[:, :, v, h] = y_qnt_blk
                x_rec_blk = xhat_blk
                # x_rec_blk_0_255 = (x_rec_blk + 0.5).mul(255).clamp_(0, 255)
                # zhat[:, :, v, h] = torch.round(x_rec_blk_0_255) / 255.0 - 0.5
                zhat[:, :, v, h] = x_rec_blk.clamp_(-0.5, 0.5)
        # arithmetic encode all blocks' symbols now
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        # encoder.encode_with_indexes(symbols_list_tobe.view(-1,1).squeeze().tolist(), indexes_list_tobe.view(-1,1).squeeze().tolist(), cdf, cdf_lengths, offsets)
        string = encoder.flush()
        return string, zhat

    def compress_blk(self, zhat, x, y_qnt, v, h, LL, RR, UU):
        # forward prediction + transform
        y_bblk = self.forward_prtr_(zhat, x)
        # get mean&scale for y
        ksi = self.get_meanscale(zhat)
        scales_blk, means_blk = ksi[:, :, v-UU:v-UU+1, h-LL:h-LL+1].chunk(2, dim=1)
        # prep indexes and symbols for arithmetic coder
        indexes = self.conditional_gaussian_model.build_indexes(scales_blk)
        y_sym_blk = self.conditional_gaussian_model.quantize(y_bblk[:, :, v-UU:v-UU+1, h-LL:h-LL+1], "symbols", means_blk)
        # inverse prediction + transform
        y_qnt_blk = y_sym_blk + means_blk
        y_qnt[:, :, v-UU, h-LL] = y_qnt_blk[:, :, 0, 0]
        xhat_bblk = self.inverse_prtr_(zhat, y_qnt)
        # return all needed
        return indexes, y_sym_blk[:, :, 0, 0], xhat_bblk[:, :, v-UU, h-LL], y_qnt_blk[:, :, 0, 0]

    def decompress(self, bitstream, LRU, xshape, chlat, devc):
        """ Decompress one image, that has been reshaped to  1 x 3*B*B x H/B x W/B,  with arithmetic decoding. """
        # prep stuff to pass to arithmetic decoder
        cdf = self.conditional_gaussian_model.quantized_cdf.tolist()
        cdf_lengths = self.conditional_gaussian_model.cdf_length.tolist()
        offsets = self.conditional_gaussian_model.offset.tolist()
        self.conditional_gaussian_model._check_cdf_size()
        self.conditional_gaussian_model._check_cdf_length()
        self.conditional_gaussian_model._check_offsets_size()
        decoder = RansDecoder()
        decoder.set_stream(bitstream)

        # get receptive field size
        L, R, U = LRU[0], LRU[1], LRU[2]
        # get outputs shape
        bt, ch, hg, wd = xshape[0], xshape[1], xshape[2], xshape[3]
        # get a reconstructed version of data which is initially all zeros
        zhat = torch.zeros((bt, ch, hg, wd), device=devc)
        y_qnt = torch.zeros((bt, chlat, hg, wd), device=devc)
        # run through model, arithmetic code, update one reconstruction pixel in zhat, do again...
        for v in range(0, hg):
            UU = max(0, v - U)
            for h in range(0, wd):
                LL = max(0, h - L)
                RR = min(wd, h + (R+1))
                # decompress_blk : ------------------------------
                # get mean&scale for y
                ksi = self.get_meanscale(zhat[:, :, UU:v+1, LL:RR])
                scales, means = ksi.chunk(2, dim=1)
                # prep indexes for arithmetic decoder, and decode from bitstream for 1 blk
                indexes = self.conditional_gaussian_model.build_indexes(scales[:, :, v-UU, h-LL])
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1)
                rv = self.conditional_gaussian_model.dequantize(rv, means[:, :, v-UU, h-LL])
                y_qnt_blk = rv.reshape(1, -1)
                # update latent with quantized blk and apply inverse transform
                y_qnt[:, :, v, h] = y_qnt_blk
                xhat = self.inverse_prtr_(zhat[:, :, UU:v+1, LL:RR], y_qnt[:, :, UU:v+1, LL:RR])
                # update zhat with reconstructed blk
                x_rec_blk = xhat[:, :, v-UU, h-LL]
                # x_rec_blk_0_255 = (x_rec_blk + 0.5).mul(255).clamp_(0, 255)
                # zhat[:, :, v, h] = torch.round(x_rec_blk_0_255) / 255.0 - 0.5
                zhat[:, :, v, h] = x_rec_blk.clamp_(-0.5, 0.5)
        # return reconstruction
        return zhat


class BlockBasedImgCompLossyNetv9(BlockBasedImgCompLossyNetv4):
    """ Same as above v5 except that
      * the auto encoder and decoder have different (also different from v8) num channels in first layers
      * the entropy model has 4 instead of 5 layers with num channels derived from M (like v7)
      * Note : the kernel sizes of encoder & decoder are different
        - au-enc layer ks : K1, K2, K3, K4
        - au-dec layer ks : K1, K4, K3, K2
    """
    def __init__(self, config):
        super(BlockBasedImgCompLossyNetv9, self).__init__(config)
        # set some parameters of the networks used in model
        B = config.block_size
        B2 = B**2
        # K1, K2, K3, K4 = config.KS[0], config.KS[1], config.KS[2], config.KS[3]  # ks for each layer
        K1, K2, K3, K4 = config.KS[0], 1, 1, 1
        P1, P2, P3, P4 = K1//2, K2//2, K3//2, K4//2                  # padding for each layer
        C1 = config.N           # num channels for each layer
        C2 = config.N // 8 * 7
        C3 = config.N // 8 * 6
        C4 = config.M
        # PREDICTION and TRANSFORM networks
        # processes x + zhat to generate y, the latent variables to send to decoder
        self.prtr_forward1 = MaskedConv2d('B', B2 * 3, C1,  1, 1, 0)  # processes x, the original block
        self.prtr_forward2 = MaskedConv2d('A', B2 * 3, C1, K1, 1, P1)  # processes zhat, neighboring blocks
        self.prtr_forward3 = nn.Sequential(                          # processes cat of previous two outputs
            GDN(C1),
            MaskedConv2d('B',     C1, C2, K2, 1, P2),
            GDN(C2),
            MaskedConv2d('B',     C2, C3, K3, 1, P3),
            GDN(C3),
            MaskedConv2d('B',     C3, C4, K4, 1, P4)
        )
        # processes zhat + y_qnt to gnerate x_rec
        self.prtr_inverse1 = MaskedConv2d('B', C4,     C1,  1, 1, 0)  # processes y_qnt, the received latent variabls of blk
        self.prtr_inverse2 = MaskedConv2d('A', B2 * 3, C1, K1, 1, P1)  # processes zhat, neighboring blocks
        self.prtr_inverse3 = nn.Sequential(                           # processes cat of previous two outputs
            GDN(C1, inverse=True),
            MaskedConv2d('B',     C1, C2, K4, 1, P4),
            GDN(C2, inverse=True),
            MaskedConv2d('B',     C2, C3, K3, 1, P3),
            GDN(C3, inverse=True),
            MaskedConv2d('B',     C3, B2*3, K2, 1, P2)
        )
        # NEIGHBORBLOCKS CONDITIONING network (generates mean+scale for conditional gaussian)
        # K1, K2, K3, K4 = config.KS[0], 1, 1, 1        # ks for each layer
        K1, K2, K3, K4 = config.KS[0], config.KS[1], 1, 1  # ks for each layer
        P1, P2, P3, P4 = K1//2, K2//2, K3//2, K4//2   # padding for each layer
        C1 = config.N // 8 * 12  # config.N // 8 * 10  # config.M * 8
        C2 = config.N // 8 * 10  # config.N // 8 * 8   # config.M * 7
        C3 = config.N // 8 * 8   # config.N // 8 * 6   # config.M * 6
        # C4 = config.N // 8 * 6      # extra
        C4 = config.M * 2  # need channels for mean and scale
        # C5 = config.M * 2  # need channels for mean and scale
        # processes zhat to generate means, stdevs
        self.get_meanscale = nn.Sequential(
            MaskedConv2d('A', B2 * 3, C1, K1, 1, P1),  # use A :zhat dreictly input
            nn.LeakyReLU(),
            MaskedConv2d('B', C1, C2, K2, 1, P2),
            nn.LeakyReLU(),
            MaskedConv2d('B', C2, C3, K3, 1, P3),
            nn.LeakyReLU(),
            MaskedConv2d('B', C3, C4, K4, 1, P4)
            # nn.LeakyReLU(),                            # extra
            # MaskedConv2d('B', C4, C5, 1, 1, 1//2)      # extra
        )
        # CDF
        self.conditional_gaussian_model = GaussianConditional(None)

    def compress(self, x, LRU, chlat,):
        """ Compress one image, that has been reshaped to  1 x 3*B*B x H/B x W/B,  with arithmetic coding. """
        # prep stuff to pass to arithmetic coder
        cdf = self.conditional_gaussian_model.quantized_cdf.tolist()
        cdf_lengths = self.conditional_gaussian_model.cdf_length.tolist()
        offsets = self.conditional_gaussian_model.offset.tolist()
        self.conditional_gaussian_model._check_cdf_size()
        self.conditional_gaussian_model._check_cdf_length()
        self.conditional_gaussian_model._check_offsets_size()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        # get receptive field size
        L, R, U = LRU[0], LRU[1], LRU[2]
        # get inputs shape
        bt, ch, hg, wd = x.shape
        # get a reconstructed version of data which is initially all zeros
        zhat = torch.zeros_like(x)
        y_qnt = torch.zeros((bt, chlat, hg, wd), device=x.device)
        # run through model, arithmetic code, update one reconstruction pixel in zhat, do again...
        for v in range(0, hg):
            UU = max(0, v - U)
            BB = min(hg, v + (U+1))
            padding_top, padding_bottom = U-v + UU, v+(U+1) - BB
            for h in range(0, wd):
                LL = max(0, h - L)
                RR = min(wd, h + (R+1))
                # pad zhat cutout if at edge of frame
                padding_left, padding_right = L-h + LL, h+(R+1) - RR
                if padding_left > 0 or padding_right > 0 or padding_top > 0 or padding_bottom > 0:
                    zhat_bblk = F.pad(zhat[:, :, UU:BB, LL:RR], (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0.0)
                else:
                    zhat_bblk = zhat[:, :, UU:BB, LL:RR]
                indexes_blk, y_sym_blk, xhat_blk, y_qnt_blk = self.compress_blk(zhat_bblk, x[:, :, v:v+1, h:h+1], v, h, LL, RR, UU)
                symbols_list.extend(y_sym_blk.squeeze().tolist())
                indexes_list.extend(indexes_blk.squeeze().tolist())
                y_qnt[:, :, v, h] = y_qnt_blk
                x_rec_blk = xhat_blk
                zhat[:, :, v, h] = x_rec_blk.clamp_(-0.5, 0.5)
        # arithmetic encode all blocks' symbols now
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        string = encoder.flush()
        return string, zhat

    def compress_blk(self, zhat, x, v, h, LL, RR, UU):
        # forward prediction + transform (note : take central 3x3 of zhat if it is larger such as 5x5)
        zW = zhat.shape[3] // 2
        y_blk = self.forward_prtr_fast(zhat[:, :, zW-1:zW+2, zW-1:zW+2], x)
        # get mean&scale for y
        ksi = self.get_meanscale_fast(zhat)
        scales_blk, means_blk = ksi.chunk(2, dim=1)
        # prep indexes and symbols for arithmetic coder
        indexes = self.conditional_gaussian_model.build_indexes(scales_blk)
        y_sym_blk = self.conditional_gaussian_model.quantize(y_blk, "symbols", means_blk)
        # inverse prediction + transform
        y_qnt_blk = y_sym_blk + means_blk
        xhat_blk = self.inverse_prtr_fast(zhat[:, :, zW-1:zW+2, zW-1:zW+2], y_qnt_blk)
        # return all needed
        return indexes, y_sym_blk[:, :, 0, 0], xhat_blk[:, :, 0, 0], y_qnt_blk[:, :, 0, 0]

    def forward_prtr_fast(self, zhat, x):
        out_x = self.prtr_forward1(x)  # processes x, the original block
        out_zhat = F.conv2d(zhat, self.prtr_forward2.weight * self.prtr_forward2.mask, self.prtr_forward2.bias, self.prtr_forward2.stride, padding=0)  # processes zhat, neighb blocks
        return self.prtr_forward3(out_x + out_zhat)

    def inverse_prtr_fast(self, zhat, y_qnt):
        out_y_qnt = self.prtr_inverse1(y_qnt)
        out_zhat = F.conv2d(zhat, self.prtr_inverse2.weight * self.prtr_inverse2.mask, self.prtr_inverse2.bias, self.prtr_inverse2.stride, padding=0)  # processes zhat, neighb bl
        return self.prtr_inverse3(out_y_qnt + out_zhat)

    def get_meanscale_fast(self, zhat):
        zout = self.get_meanscale[1].forward(F.conv2d(zhat, self.get_meanscale[0].weight * self.get_meanscale[0].mask,
                                                      self.get_meanscale[0].bias, self.get_meanscale[0].stride, padding=0))
        zout = self.get_meanscale[3].forward(F.conv2d(zout, self.get_meanscale[2].weight * self.get_meanscale[2].mask,
                                                      self.get_meanscale[2].bias, self.get_meanscale[2].stride, padding=0))
        zout = self.get_meanscale[5].forward(F.conv2d(zout, self.get_meanscale[4].weight * self.get_meanscale[4].mask,
                                                      self.get_meanscale[4].bias, self.get_meanscale[4].stride, padding=0))
        zout = F.conv2d(zout, self.get_meanscale[6].weight * self.get_meanscale[6].mask,
                        self.get_meanscale[6].bias, self.get_meanscale[6].stride, padding=0)
        return zout

    def decompress(self, bitstream, LRU, xshape, chlat, devc):
        """  Decompress one image, that has been reshaped to  1 x 3*B*B x H/B x W/B,  with arithmetic decoding.  """
        # prep stuff to pass to arithmetic decoder
        cdf = self.conditional_gaussian_model.quantized_cdf.tolist()
        cdf_lengths = self.conditional_gaussian_model.cdf_length.tolist()
        offsets = self.conditional_gaussian_model.offset.tolist()
        self.conditional_gaussian_model._check_cdf_size()
        self.conditional_gaussian_model._check_cdf_length()
        self.conditional_gaussian_model._check_offsets_size()
        decoder = RansDecoder()
        decoder.set_stream(bitstream)

        # get receptive field size
        L, R, U = LRU[0], LRU[1], LRU[2]
        # get outputs shape
        bt, ch, hg, wd = xshape[0], xshape[1], xshape[2], xshape[3]
        # get a reconstructed version of data which is initially all zeros
        zhat = torch.zeros((bt, ch, hg, wd), device=devc)
        y_qnt = torch.zeros((bt, chlat, hg, wd), device=devc)
        # run through model, arithmetic code, update one reconstruction pixel in zhat, do again...
        for v in range(0, hg):
            UU = max(0, v - U)
            BB = min(hg, v + (U+1))
            padding_top, padding_bottom = U-v + UU, v+(U+1) - BB
            for h in range(0, wd):
                LL = max(0, h - L)
                RR = min(wd, h + (R+1))
                # pad zhat cutout if at edge of frame
                padding_left, padding_right = L-h + LL, h+(R+1) - RR
                if padding_left > 0 or padding_right > 0 or padding_top > 0 or padding_bottom > 0:
                    zhat_bblk = F.pad(zhat[:, :, UU:BB, LL:RR], (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0.0)
                else:
                    zhat_bblk = zhat[:, :, UU:BB, LL:RR]
                # decompress_blk : ------------------------------
                # get mean&scale for y
                ksi = self.get_meanscale_fast(zhat_bblk)
                scales, means = ksi.chunk(2, dim=1)
                # prep indexes for arithmetic decoder, and decode from bitstream for 1 blk
                indexes = self.conditional_gaussian_model.build_indexes(scales[:, :, 0, 0])
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1)
                rv = self.conditional_gaussian_model.dequantize(rv, means[:, :, 0, 0])
                y_qnt_blk = rv.reshape(1, -1)
                # update latent with quantized blk and apply inverse transform
                zW = zhat_bblk.shape[3] // 2
                xhat_blk = self.inverse_prtr_fast(zhat_bblk[:, :, zW - 1:zW + 2, zW - 1:zW + 2], y_qnt_blk.unsqueeze(dim=2).unsqueeze(dim=3))
                # update zhat with reconstructed blk
                x_rec_blk = xhat_blk[:, :, 0, 0]
                # x_rec_blk_0_255 = (x_rec_blk + 0.5).mul(255).clamp_(0, 255)
                # zhat[:, :, v, h] = torch.round(x_rec_blk_0_255) / 255.0 - 0.5
                zhat[:, :, v, h] = x_rec_blk.clamp_(-0.5, 0.5)
        # return reconstruction
        return zhat


class BlkBasedPostProcessing(nn.Module):
    def __init__(self, config):
        super(BlkBasedPostProcessing, self).__init__()
        # set some parameters of the networks used in model
        B = config.block_size
        B2 = B**2
        C1 = B2 * 3
        C2 = C1 * 4
        self.res_net = nn.Sequential(
            nn.Conv2d(in_channels=C1, out_channels=C2, kernel_size=3, stride=1, padding=0),  # make padding zero to avoid filtering image boundaries
            nn.LeakyReLU(),  # nn.Tanh(),
            # nn.Conv2d(in_channels=C2, out_channels=C2, kernel_size=3, stride=1, padding=3//2),
            # nn.LeakyReLU(),  # nn.Tanh(),
            nn.Conv2d(in_channels=C2, out_channels=C1, kernel_size=1, stride=1, padding=1//2)
        )
        # self.res_net = nn.Sequential(
        #     nn.Conv2d(in_channels=C1, out_channels=C1, kernel_size=3, stride=1, padding=0),
        # )

    def forward(self, x):
        x_res = self.res_net(x)
        return x + F.pad(x_res, (1, 1, 1, 1), "constant", 0)   # (1, 1, 2, 2) pad last dim by (1, 1) and 2nd to last by (2, 2)
