import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
# from visdom import Visdom
import logging
from agents.base import BaseAgent
from graphs.models.BlockBasedImgCompLossy_net import BlockBasedImgCompLossyNetv4, \
    BlockBasedImgCompLossyNetv9, BlkBasedPostProcessing
from graphs.losses.rate_dist import TrainRDLoss, TrainDLoss
from dataloaders.image_dl_ACL import ImageDataLoader_ACL
from loggers.rate import RateLogger, RDLogger
from utils.image_plots import display_image_in_actual_size, plot_hist_of_rgb_image
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as amp_autocast
import math
import time
from pytorch_msssim import ms_ssim
# from torchsummary import summary


class BlockBasedImgCompLossyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.block_size = config.block_size
        if config.net_version == "v4":
            self.model0 = BlockBasedImgCompLossyNetv4(config)  
        elif config.net_version == "v9":
            self.model0 = BlockBasedImgCompLossyNetv9(config)
        else:
            self.model0 = None
        self.model0 = self.model0.to(self.device)
        self.use_postpm = config.use_postpm
        if self.use_postpm:
            self.postpm = BlkBasedPostProcessing(config)
            self.postpm = self.postpm.to(self.device)
        # self.model1 = BlockBasedImgCompLossyNetv4(config)
        # self.model1 = self.model1.to(self.device)
        # self.model2 = BlockBasedImgCompLossyNetv4(config)
        # self.model2 = self.model2.to(self.device)
        self.use_amp = config.use_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        self.lr = self.config.learning_rate
        self.optimizer = optim.Adam([{'params': self.model0.parameters(), 'lr':self.lr}])
        if self.use_postpm:
            self.optimizer_pp = optim.Adam([{'params': self.postpm.parameters(), 'lr':self.lr}])
        # self.optimizer = optim.Adam(
        #     [{'params': self.model0.parameters(), 'lr':self.lr},
        #      {'params': self.model1.parameters(), 'lr':self.lr},
        #      {'params': self.model2.parameters(), 'lr':self.lr}]
        # )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=4,
                                                              threshold=0.0003, threshold_mode='rel',
                                                              cooldown=1, min_lr=4e-05, eps=1e-08, verbose=False)  # !!! 4e-05
        self.chained_training = config.chained_training
        self.acl_lr_alpha = config.acl_lr_alpha
        self.convergence_decsn_mode = config.convergence_decsn_mode
        self.grad_acc_iters = config.grad_acc_iters
        self.loss_prnt_iters = config.loss_prnt_iters
        self.data_loader = ImageDataLoader_ACL(config)
        self.lambda_ = config.lambda_
        self.distortion= config.distortion
        self.loss_switch_thr = config.loss_switch_thr
        self.training_loss_switch = config.training_loss_switch
        if self.training_loss_switch == 0:
            self.train_loss = TrainDLoss(config.lambda_, distortion=self.distortion, blocksize=self.block_size)
            print("Starting training with only lambda*Distortion training loss...")
        else:
            self.train_loss = TrainRDLoss(config.lambda_, distortion=self.distortion, blocksize=self.block_size)
        self.valid_loss = TrainRDLoss(config.lambda_, distortion=self.distortion, blocksize=self.block_size)
        self.train_logger0 = RDLogger(distortion=self.distortion)
        self.trnit_logger0 = RDLogger(distortion=self.distortion)  # to report for every 1000 iterations inside and epoch
        self.train_logger1 = RDLogger(distortion=self.distortion)
        self.trnit_logger1 = RDLogger(distortion=self.distortion)
        # self.train_logger2 = RDLogger()
        # self.trnit_logger2 = RDLogger()
        self.valid_logger0 = RDLogger(distortion=self.distortion)
        # self.valid_logger1 = RDLogger()
        # self.valid_logger2 = RDLogger()
        self.rcrec_logger  = RDLogger(distortion=self.distortion)
        self.gents_logger  = RDLogger(distortion=self.distortion)
        self.train_logger_pp = RDLogger(distortion=self.distortion)
        # self.test_logger = RDLogger()
        # self.viz = Visdom(raise_exceptions=True)
        if config.mode in ['test', 'validate', 'validate_recu_reco', 'validate_recu_reco_fast', 
                           'gen_train_set', 'gen_train_set_postproc', 
                           'update_model', 'eval_model', 'model_size']:
            self.load_checkpoint(self.config.modelbest_file_load)
        elif config.resume_training and config.mode in ['train_all_acl']:
            self.load_checkpoint(self.config.checkpoint_file)
            # self.optimizer.param_groups[0]['lr'] = self.lr
            # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=4,
            #                                                       threshold=0.0003, threshold_mode='rel',
            #                                                       cooldown=1, min_lr=1e-05, eps=1e-08, verbose=False)  # 5e-5
            # self.lr_flag = 1
        elif config.resume_training == False:
            #self.load_checkpoint(self.config.modelbest_file_load)
            #self.optimizer.param_groups[0]['lr'] = self.lr
            print('No loading checkpoint with a model. Model will be trained from scratch.')
        # freeze parameters in some parts of model
        # self.model0.freeze_some_network("aenc", config.freeze_aenc)
        # self.model0.freeze_some_network("adec", config.freeze_adec)
        # self.model0.freeze_some_network("emdl", config.freeze_emdl)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def train_all_acl(self):
        acl_itr = self.config.acl_itr
        acl_itr0_rdloss_thr = self.config.acl_itr0_rdloss_threshold
        while True:  # loop over ACL ITERATIONS
            # init best losses to 0 for each acl itr so you can store a model for each acl itr
            #self.best_validrr_loss = float('inf')
            #self.best_valid_loss = float('inf')
            self.logger.info(" ")
            self.logger.info("*** ACL itr " + str(acl_itr) + " started ***")
            cnt_no_improvement_valid, cnt_no_improvement_validrr = 0, 0
            if self.convergence_decsn_mode == 'ol_loss':
                self.best_valid_loss = float('inf')
                # self.best_validrr_loss = float('inf')
                if acl_itr < len(self.config.cnt_no_improvement_valid_thresholds):
                    cnt_no_improvement_valid_threshold = self.config.cnt_no_improvement_valid_thresholds[acl_itr]
                else:
                    cnt_no_improvement_valid_threshold = self.config.cnt_no_improvement_valid_thresholds[-1]
                self.logger.info(
                    "This acl itr's count threshold for no improvement validations (i.e. convergence) : {}".format(cnt_no_improvement_valid_threshold))
            elif self.convergence_decsn_mode == 'cl_loss':
                if acl_itr < len(self.config.cnt_no_improvement_valid_thresholds):
                    cnt_no_improvement_validrr_threshold = self.config.cnt_no_improvement_valid_thresholds[acl_itr]
                else:
                    cnt_no_improvement_validrr_threshold = self.config.cnt_no_improvement_valid_thresholds[-1]
                self.logger.info(
                    "This acl itr's count threshold for no improvement rr validations (i.e. convergence) : {}".format(cnt_no_improvement_validrr_threshold))
            if acl_itr == 0:
                self.logger.info("This acl itr's (0th) loss threshold for open loop loss : {:.6f}".format(acl_itr0_rdloss_thr))
            self.logger.info("This acl itr's convergence decision will be made with '{}'".format(self.convergence_decsn_mode))
            self.logger.info("This acl itr's deep learning training will be with self.chained_training = {}".format(self.chained_training))
            valid_loss = float('inf')
            change_convergence_decsn_mode = False
            while True:  # loop over DEEP LEARNING EPOCHS
                self.train_one_epoch()
                # Save best valid and validrr results
                is_best, is_bestrr = False, False
                if not (self.current_epoch+1) % self.config.validate_every:
                    valid_loss = self.validate()
                    is_best = valid_loss < self.best_valid_loss
                    if is_best:
                        self.best_valid_loss = valid_loss
                        cnt_no_improvement_valid = 0
                    else:
                        cnt_no_improvement_valid += 1
                    self.save_checkpoint(is_best=is_best, acl_itr=acl_itr, rr=False)
                if not (self.current_epoch+1) % max(1, self.config.validate_recu_reco_every - acl_itr//1):
                    validrr_loss = self.validate_recu_reco()
                    is_bestrr = validrr_loss < self.best_validrr_loss
                    if is_bestrr:
                        self.best_validrr_loss = validrr_loss
                        cnt_no_improvement_validrr = 0
                    else:
                        cnt_no_improvement_validrr += 1
                    self.save_checkpoint(is_best=is_bestrr, acl_itr=acl_itr, rr=True)
                self.current_epoch += 1
                # Did DL training converge ? Finish loop over DL epochs ?
                if self.convergence_decsn_mode == 'ol_loss':
                    if (acl_itr > 0 and cnt_no_improvement_valid > cnt_no_improvement_valid_threshold) or (acl_itr == 0 and valid_loss < acl_itr0_rdloss_thr):
                        self.logger.info("Declaring converge for this ACL itr. " +
                                         "cnt_no_improvement_valid={} > cnt_no_improvement_valid_threshold={}. "
                                         .format(cnt_no_improvement_valid, cnt_no_improvement_valid_threshold))
                        # if acl_itr > (2+0) and not (self.best_valid_loss < 0.99 * self.prev_aclitr_best_valid_loss):  # change mode if ol_loss improvement not enough
                        if acl_itr >= (2+0) and not (self.best_validrr_loss < 0.99 * self.prev_aclitr_best_validrr_loss):
                            change_convergence_decsn_mode = True
                        break
                elif self.convergence_decsn_mode == 'cl_loss':
                    if cnt_no_improvement_validrr > cnt_no_improvement_validrr_threshold:
                        self.logger.info("Declaring converge for this ACL itr ({}). ".format(acl_itr) +
                                         "cnt_no_improvement_validrr={} > cnt_no_improvement_validrr_threshold={}. "
                                         .format(cnt_no_improvement_validrr, cnt_no_improvement_validrr_threshold))
                        break
            # print best validation losses of this acl iteration; choose model to use as initialization for next acl itr
            self.logger.info(" ++ Best loss for this acl itr's open loop:{:.6f}".format(self.best_valid_loss))
            self.logger.info(" ++ Best loss for this acl itr's closed loop (rr):{:.6f}".format(self.best_validrr_loss))
            self.logger.info("*** ACL itr " + str(acl_itr) + " finished ***")
            # finish all acl iterations ??? (just finish all_acl manually...)
            # generate training set and validation set for next acl iteration (reconstructed images)
            self.logger.info(" -- Will use the best model from previous ACL itr (loading corresponding checkpoint below) to")
            self.logger.info("     a) obtain training and validation sets for next acl iteration")
            self.logger.info("     b) as initial point for the training in the next acl iteration.")
            self.config.mode = "gen_train_set"
            self.config.acl_itr = acl_itr
            self.data_loader = ImageDataLoader_ACL(self.config)
            if self.convergence_decsn_mode == 'ol_loss' and change_convergence_decsn_mode is False:
                self.load_checkpoint("model_best_acl_" + str(acl_itr - 0) + ".pth.tar")  # note: if this file does not exist, will continue with current model parameters
                # self.load_checkpoint("model_best_acl_" + str(acl_itr - 0) + "rr.pth.tar")
                self.prev_aclitr_best_valid_loss = self.best_valid_loss
                self.prev_aclitr_best_validrr_loss = self.best_validrr_loss
            elif self.convergence_decsn_mode == 'cl_loss' or change_convergence_decsn_mode is True:
                self.load_checkpoint("model_best_acl_" + str(acl_itr - 0) + "rr.pth.tar") # note: if this file does not exist, will continue with current model parameters
                if change_convergence_decsn_mode is True:
                    self.convergence_decsn_mode = 'cl_loss'
                    self.chained_training = True
                    # self.config.batch_size *= 2
                    self.optimizer.param_groups[0]['lr'] *= 0.66
                    self.logger.info("NOTE : Changing acl iteration's convergence decision mode to '{}', ".format(self.convergence_decsn_mode) +
                                     "making self.chained_training={}, and".format(self.chained_training) +
                                     " doubling batch_size ({}) for next acl iterations.".format(self.config.batch_size))
            self.logger.info("Will now start generating training and validation sets for next acl iteration:")
            self.generate_training_set_next_acl_itr(self.data_loader.train_loader)  # training set
            self.generate_training_set_next_acl_itr(self.data_loader.valid_loader)  # validation set
            # prepare for next acl iteration
            acl_itr = acl_itr + 1
            self.config.mode = "train_all_acl"
            self.config.acl_itr = acl_itr
            self.data_loader = ImageDataLoader_ACL(self.config)
            # update lr and whether chained_training for next acl itr   !!!! ????
            # if self.optimizer.param_groups[0]['lr'] > 0.45e-4:
            #     self.optimizer.param_groups[0]['lr'] *= self.acl_lr_alpha
            #     self.logger.info("Updated learning rate for next acl iteration to {:.6f}".format(self.optimizer.param_groups[0]['lr']))
            w1_lr = max(5 - acl_itr, 0) / 10
            lr_next_acl_itr = self.lr * w1_lr + self.optimizer.param_groups[0]['lr'] * (1.0 - w1_lr)
            min_lr_ = (4e-5 if self.convergence_decsn_mode == 'ol_loss' else 2e-05)
            self.optimizer = optim.Adam([{'params': self.model0.parameters(), 'lr': lr_next_acl_itr}])
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.8, patience=4,
                                                                  threshold=0.0003, threshold_mode='rel',
                                                                  cooldown=1, min_lr=min_lr_, eps=1e-08, verbose=False)

    def train_one_epoch(self):
        self.model0.train()
        # self.model1.train()
        # self.model2.train()
        for batch_idx, (x, zhat) in enumerate(self.data_loader.train_loader):
            # get the original images/data
            x = x.to(self.device)
            x = (x - 0.5) * 1
            # get the reconstructed/noisy version (i.e. zhat) of images/data
            zhat = zhat.to(self.device)
            zhat = (zhat - 0.5) * 1
            # arrange pixels inside a block to channel dimension (Note : make sure patch_Size is multiple of B)
            if self.block_size > 1:
                x = arrange_block_pixels_to_channel_dim(x, self.block_size, self.device)
                zhat = arrange_block_pixels_to_channel_dim(zhat, self.block_size, self.device)
            # run through model, calculate loss, back-prop etc.
            with amp_autocast(enabled=self.use_amp):
                xhat0, self_infos0 = self.model0(zhat, x)
                rd_loss0, mse_loss0, rate_loss0 = self.train_loss.forward(x, xhat0, self_infos0)
                rd_loss0 /= self.grad_acc_iters
                if self.chained_training is True:
                    xhat1, self_infos1 = self.model0(xhat0, x)
                    rd_loss1, mse_loss1, rate_loss1 = self.train_loss.forward(x, xhat1, self_infos1)
                    rd_loss1 /= self.grad_acc_iters
                    rd_loss = rd_loss0 * 0.5 + rd_loss1 * 0.5   # change weights upto 0.15, 0.85 !?
                elif self.chained_training is False:
                    rd_loss = rd_loss0
            # xhat2, self_infos2 = self.model0(xhat1, x)
            # xhat1, self_infos1 = self.model1(xhat0, x)    # !!!! extra stuff here in in line below to have recu reco simulation
            # xhat2, self_infos2 = self.model2(xhat1, x)  # !!!!
            # rd_loss2, mse_loss2, rate_loss2 = self.train_loss.forward(x, xhat2, self_infos2)
            # get convergence loss for xhat and zha to be equal or ismilar
            # mse_loss01 = self.mse_loss(xhat0, xhat1)
            # mse_loss12 = self.mse_loss(xhat1, xhat2)
            # conv_loss = (mse_loss01 + mse_loss12) / 2 * self.lambda_
            # cnvg_mse = self.mse_loss(xhat0, zhat)
            # cnvg_loss = self.cnvg_lmbd * self.lambda_ * cnvg_mse
            # rate_loss3 = torch.zeros_like(rate_loss2)
            # rd_loss3 = mse_loss3 * self.lambda_   # ! error between two reconstructions, try to match them
            # rd_loss3, mse_loss3, rate_loss3 = self.train_loss.forward(x, xhat3, self_infos3)
            # ((rd_loss0 * 0.333 + rd_loss1 * 0.333 + rd_loss2 * 0.334 + conv_loss * 0.8) / self.grad_acc_iters).backward()
            # ((rd_loss0 + cnvg_loss) / self.grad_acc_iters).backward()
            self.amp_scaler.scale(rd_loss).backward()
            # (rd_loss1 / self.grad_acc_iters).backward()
            # ((rd_loss0 * 0.33 + rd_loss1 * 0.33 + rd_loss1 * 0.34) / self.grad_acc_iters).backward()
            # gradeint accumulation of grad_acc_iters
            if ((self.current_iteration + 1) % self.grad_acc_iters == 0) or ((batch_idx+1) == len(self.data_loader.train_loader)):
                # apply gradient clipping/scaling (if loss has been switched to R+lD)
                if self.training_loss_switch == 1:
                    # Unscales the gradients of optimizer's assigned params in-place
                    self.amp_scaler.unscale_(self.optimizer)
                    # https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
                    nn.utils.clip_grad_value_(self.model0.parameters(), clip_value=0.5) # 0.1  0.99
                    # nn.utils.clip_grad_value_(self.model1.parameters(), clip_value=0.99)  # 0.1
                    # nn.utils.clip_grad_value_(self.model2.parameters(), clip_value=0.99)  # 0.1
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01, norm_type=2)
                # update weights, iteration number and log losses
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
                self.optimizer.zero_grad()
            self.current_iteration += 1
            self.train_logger0(rd_loss0.item(), mse_loss0.item(), rate_loss0.item())
            ##self.trnit_logger0(rd_loss0.item(), mse_loss0.item(), rate_loss0.item())
            if self.chained_training is True:
                self.train_logger1(rd_loss1.item(), mse_loss1.item(), rate_loss1.item())
                ##self.trnit_logger1(rd_loss1.item(), mse_loss1.item(), rate_loss1.item())
            # self.train_logger2(rd_loss2.item(), mse_loss2.item(), rate_loss2.item())
            # self.trnit_logger2(rd_loss2.item(), mse_loss2.item(), rate_loss2.item())
            if (self.current_iteration + 1) % self.loss_prnt_iters == 0:
                ##trnit_rd_loss0, trnit_mse_loss0, trnit_rate_loss0, _ = self.trnit_logger0.display(lr=self.optimizer.param_groups[0]['lr'], typ='it')
                ##if self.chained_training is True:
                ##    trnit_rd_loss1, trnit_mse_loss1, trnit_rate_loss1, _ = self.trnit_logger1.display(lr=self.optimizer.param_groups[0]['lr'], typ='it')
                # trnit_rd_loss2, trnit_mse_loss2, trnit_rate_loss2, _ = self.trnit_logger2.display(lr=self.optimizer.param_groups[0]['lr'], typ='it')
                # switch to R+lD loss once training mse for an epoch drops below threshold
                # if 10.0 < self.loss_switch_thr and self.training_loss_switch == 0:
                if rd_loss0 < self.loss_switch_thr and self.training_loss_switch == 0:
                    self.train_loss = TrainRDLoss(self.lambda_)
                    self.logger.info("Switching training loss to Rate+lambda*Distortion (it was only lambda*Distortion up to here)")
                    self.training_loss_switch = 1
        train_rd_loss0, train_mse_loss0, train_rate_loss0, _ = self.train_logger0.display(lr=self.optimizer.param_groups[0]['lr'], typ='tr')
        if self.chained_training is True:
            train_rd_loss1, train_mse_loss1, train_rate_loss1, _ = self.train_logger1.display(lr=self.optimizer.param_groups[0]['lr'], typ='tr')
        # train_rd_loss2, train_mse_loss2, train_rate_loss2, _ = self.train_logger2.display(lr=self.optimizer.param_groups[0]['lr'], typ='tr')

    def train_postproc_mdl(self):
        self.model0.eval()
        # turn off gradients for actual compression model
        for name, p in self.model0.named_parameters():
            p.requires_grad = False
        # train post proc module for maultiple epochs
        train_mse_loss_pp_best = float('inf')
        cont_training = True
        while cont_training:
            # TRAIN post proc module for one epoch
            for batch_idx, (x, x_recureco) in enumerate(self.data_loader.train_loader):
                # get the original images/data
                x = x.to(self.device)
                x = (x - 0.5) * 1
                # get the reconstructed/noisy version (i.e. zhat) of images/data
                x_recureco = x_recureco.to(self.device)
                x_recureco = (x_recureco - 0.5) * 1
                # arrange pixels inside a block to channel dimension (Note : make sure patch_size is multiple of B)
                if self.block_size > 1:
                    x = arrange_block_pixels_to_channel_dim(x, self.block_size, self.device)
                    x_recureco = arrange_block_pixels_to_channel_dim(x_recureco, self.block_size, self.device)
                # check images
                #display_image_in_actual_size(x, self.block_size, self.device)
                #display_image_in_actual_size(x_recureco, self.block_size, self.device)
                # run through model, calculate loss, back-prop etc.
                x_postpd = self.postpm(x_recureco)
                loss_mse = self.mse_loss(x, x_postpd)
                loss_mse.backward()
                self.optimizer_pp.step()
                self.optimizer_pp.zero_grad()
                #self.current_iteration += 1
                self.train_logger_pp(0.0, loss_mse.item(), 0.0)
            _, train_mse_loss_pp, _, _ = self.train_logger_pp.display(lr=self.optimizer_pp.param_groups[0]['lr'], typ='tr')
            # continue TRAINING for another epoch ?
            if train_mse_loss_pp > 0.999 * train_mse_loss_pp_best:
                cont_training = False
            if train_mse_loss_pp < train_mse_loss_pp_best:
                train_mse_loss_pp_best = train_mse_loss_pp
        # VALIDATE now entire model according to validate_recu_reco_fast
        self.validate_recu_reco_fast(bool_postprocess=True)

    @torch.no_grad()
    def validate(self):
        self.model0.eval()
        # self.model1.eval()
        # self.model2.eval()
        with torch.no_grad():
            mse_losses = []
            for batch_idx, (x, zhat) in enumerate(self.data_loader.valid_loader):
                # get the original images/data
                x = x.to(self.device)
                x = (x - 0.5) * 1
                # get the reconstructed/noisy version (i.e. zhat) of images/data
                zhat = zhat.to(self.device)
                zhat = (zhat - 0.5) * 1
                # arrange pixels inside a block to channel dimension (Note : make sure patch_Size is multiple of B)
                if self.block_size > 1:
                    x = arrange_block_pixels_to_channel_dim(x, self.block_size, self.device)
                    zhat = arrange_block_pixels_to_channel_dim(zhat, self.block_size, self.device)
                # run through model, calculate loss
                xhat0, self_infos0 = self.model0(zhat, x)
                xhat0.clamp_(-0.5, 0.5)
                # xhat1, self_infos1 = self.model1(xhat0, x)
                # xhat1.clamp_(-0.5, 0.5)
                # xhat2, self_infos2 = self.model2(xhat1, x)
                # xhat2.clamp_(-0.5, 0.5)
                rd_loss0, mse_loss0, rate_loss0 = self.train_loss.forward(x, xhat0, self_infos0)
                # rd_loss1, mse_loss1, rate_loss1 = self.train_loss.forward(x, xhat1, self_infos1)
                # rd_loss2, mse_loss2, rate_loss2 = self.train_loss.forward(x, xhat2, self_infos2)
                # cnvg_mse = self.mse_loss(xhat0, zhat)
                # cnvg_loss = self.cnvg_lmbd * self.lambda_ * cnvg_mse
                self.valid_logger0(rd_loss0.item(), mse_loss0.item(), rate_loss0.item())
                # self.valid_logger1(cnvg_loss.item(), cnvg_mse.item(), 0.0)
                # self.valid_logger2(rd_loss2.item(), mse_loss2.item(), rate_loss2.item())
                mse_losses.append(mse_loss0.item())
                # Plot reconstructed images ?
                if False:
                    display_image_in_actual_size(x, self.block_size, self.device) # original img
                    display_image_in_actual_size(zhat, self.block_size, self.device)  # neighbor blocks img
                    display_image_in_actual_size(xhat0, self.block_size, self.device) # reconstructed img
                    # display_image_in_actual_size(xhat1, self.block_size, self.device) # reconstructed img
                    # display_image_in_actual_size(xhat2, self.block_size, self.device) # reconstructed img
            valid_rd_loss0, valid_mse_loss0, valid_rate_loss0, _ = self.valid_logger0.display(lr=0.0, typ='va')
            # valid_rd_loss1, valid_mse_loss1, valid_rate_loss1, _ = self.valid_logger1.display(lr=0.0, typ='va')
            # valid_rd_loss2, valid_mse_loss2, valid_rate_loss2, _ = self.valid_logger2.display(lr=0.0, typ='va')
            ##self.logger.info(f'avg_psnr = {10.0 * torch.log10(1.0 / torch.tensor(mse_losses)).mean().item():.2f}')
            # if self.lr_flag == 1:
            #     self.optimizer.param_groups[0]['lr'] = self.lr
            #     self.lr_flag = 0
            self.scheduler.step(valid_rd_loss0) # for
            # self.scheduler.step()
            # return valid_rd_loss0 * 0.333 + valid_rd_loss1 * 0.333 + valid_rd_loss2 * 0.334
            # return valid_rd_loss0 + cnvg_loss
            return valid_rd_loss0

    @torch.no_grad()
    def validate_recu_reco(self):
        self.model0.eval()   # !!! Caution : Which model to use in recursive reconstruction ? Also changemodel number below !!!
        save_blkbsd_rdcosts_to_disk = False  # make this False if you don't know what it does
        if save_blkbsd_rdcosts_to_disk:
            list_rdcost_tensors_per_img = []
            list_images = []
        with torch.no_grad():
            mse_losses = []
            for batch_idx, (x, zhat) in enumerate(self.data_loader.valid_loader):
                x = x.to(self.device)
                # shift pixels to -0.5, 0.5 range
                x = (x - 0.5) * 1
                # !! NOTE : zhat used here for debugging.. comment later.. get the reconstructed/noisy version (i.e. zhat) of images/data
                # zhat = zhat.to(self.device)
                # zhat = (zhat - 0.5) * 1
                # arrange pixels inside a block to channel dimension (Note : make sure patch_Size is multiple of B)
                if self.block_size > 1:
                    x = arrange_block_pixels_to_channel_dim(x, self.block_size, self.device)
                    # zhat = arrange_block_pixels_to_channel_dim(zhat, self.block_size, self.device)
                # get inputs shape
                bt, ch, hg, wd = x.shape
                # cut input into tiles and put them in batch dimension
                TS = 16  # tile size
                x_tiles = torch.zeros(int(hg*wd/TS**2), ch, TS, TS, device=x.device)
                for v in range(0, hg//TS):
                    for h in range(0, wd//TS):
                        x_tiles[v*(wd//TS)+h, :, :, :] = x[:, :, v*TS:v*TS+TS, h*TS:h*TS+TS]
                # get a reconstructed version of data which is initially all zeros
                zhat_tiles = torch.zeros_like(x_tiles)
                # run through model, calculate loss, update one reconstruction pixel in zhat, do again...
                for v in range(0, TS):
                    # print('{:d} '.format(v), end='')  # print which line is being coded
                    for h in range(0, TS):
                        xhat_tiles, self_infos = self.model0(zhat_tiles, x_tiles)
                        # xhat_tiles.clamp_(-0.5, 0.5)
                        # if h < (2*4*0) or v < (2*4*0):
                        #     zhat_tiles[:, :, v, h] = x_tiles[:, :, v, h]
                        # else:
                        #     zhat_tiles[:, :, v, h] = xhat_tiles[:, :, v, h]
                        x_rec_blk = xhat_tiles[:, :, v, h]
                        x_rec_blk_0_255 = (x_rec_blk + 0.5).mul(255).clamp_(0, 255)
                        zhat_tiles[:, :, v, h] = torch.round(x_rec_blk_0_255) / 255.0 - 0.5
                # put reconstruction in tile format back to regular format
                xhat = torch.zeros_like(x)
                for v in range(0, hg//TS):
                    for h in range(0, wd//TS):
                        xhat[:, :, v*TS:v*TS+TS, h*TS:h*TS+TS] = xhat_tiles[v*(wd//TS)+h, :, :, :]
                # !!! clip xhat to [-0.5 to 0.5], this might give better mse (plotting it gives warnings for clipping)
                xhat.clamp_(-0.5, 0.5)  # this in place version of xhat = torch.clamp(xhat, -0.5, 0.5)
                rd_loss, mse_loss, rate_loss = self.valid_loss(x, xhat, self_infos)
                self.rcrec_logger(rd_loss.item(), mse_loss.item(), rate_loss.item())
                mse_losses.append(mse_loss.item())
                # Plot reconstructed images ?
                if False:
                    display_image_in_actual_size(x, self.block_size, self.device) # original img
                    display_image_in_actual_size(xhat, self.block_size, self.device) # reconstructed img
                if save_blkbsd_rdcosts_to_disk:
                    bits_per_blk = torch.zeros(bt, self_infos.shape[1], hg, wd, device=x.device)
                    for v in range(0, hg//TS):
                        for h in range(0, wd//TS):
                            bits_per_blk[:, :, v*TS:v*TS+TS, h*TS:h*TS+TS] = self_infos[v*(wd//TS)+h, :, :, :]
                    mseloss = nn.MSELoss(reduction='none')
                    mse_x_xhat = mseloss(x, xhat)
                    rdcost_per_blk = bits_per_blk.sum(dim=1) + self.config.lambda_ * mse_x_xhat.sum(dim=1)
                    list_rdcost_tensors_per_img.append(rdcost_per_blk)
                    x01_HW = arrange_channel_dim_to_block_pixels(x +0.5, self.block_size, dev=x.device)
                    list_images.append(x01_HW)
            if save_blkbsd_rdcosts_to_disk:
                filename1 = f"list_rdcost_tensors_per_blk_B{self.block_size}_{self.config.lambda_}.pt"
                filename2 = f"list_orig_images_B{self.block_size}_{self.config.lambda_}.pt"
                torch.save(list_rdcost_tensors_per_img, self.config.out_dir + filename1)
                torch.save(list_images, self.config.out_dir + filename2)
            valid_rd_loss, valid_mse_loss, valid_rate_loss, _ = self.rcrec_logger.display(lr=0.0, typ='va')
            ##self.logger.info(f'avg_psnr = {10.0 * torch.log10(1.0 / torch.tensor(mse_losses)).mean().item():.2f}')
            # self.scheduler.step(valid_rd_loss)
            return valid_rd_loss

    def get_lru_(self, KS, mode='validation'):
        # LRU = int((torch.Tensor(self.config.KS) // 2).sum())      # but we have also the decoder !
        # LRU = int((torch.Tensor(self.config.KS) // 2).sum()) * 2  # but
        if mode == 'validation':
            LRU = int((torch.Tensor(self.config.KS) // 2).sum()) + int((torch.Tensor(self.config.KS[1:]) // 2).sum())  # enc+dec
        elif mode == 'compress' or mode == 'decompress':
            LRU = int((torch.Tensor(self.config.KS) // 2).sum())  # enc or dec
        L, R, U = LRU, LRU, LRU
        return L, R, U

    @torch.no_grad()
    def validate_recu_reco_fast(self, bool_postprocess=False):
        self.model0.eval()   # !!! Caution : Which model to use in recursive reconstruction ? Also changemodel number below !!!
        with torch.no_grad():
            # get receptive field size
            L, R, U = self.get_lru_(self.config.KS)
            # list to store each pictures mse individually
            mse_losses = []
            for batch_idx, (x, zhat) in enumerate(self.data_loader.valid_loader):
                x = x.to(self.device)
                # shift pixels to -0.5, 0.5 range
                x = (x - 0.5) * 1
                # arrange pixels inside a block to channel dimension (Note : make sure patch_Size is multiple of B)
                if self.block_size > 1:
                    x = arrange_block_pixels_to_channel_dim(x, self.block_size, self.device)
                # get inputs shape
                bt, ch, hg, wd = x.shape
                # get a reconstructed version of data which is initially all zeros
                zhat = torch.zeros_like(x)
                self_infos = torch.zeros(bt, self.config.M, hg, wd, device=x.device)  # note the channels
                # run through model, calculate loss, update one reconstruction pixel in zhat, do again...
                for v in range(0, hg):
                    # print('{:d} '.format(v), end='')  # print which line is being coded
                    for h in range(0, wd):
                        LL = max(0, h - L)
                        RR = min(wd, h + (R+1))
                        UU = max(0, v - U)
                        xhat_tmp, self_infos_tmp = self.model0(zhat[:, :, UU:v+1, LL:RR], x[:, :, UU:v+1, LL:RR])
                        self_infos[:, :, v, h] = self_infos_tmp[:, :, v-UU, h-LL]
                        x_rec_blk = xhat_tmp[:, :, v-UU, h-LL]
                        # x_rec_blk_0_255 = (x_rec_blk + 0.5).mul(255).clamp_(0, 255)
                        # zhat[:, :, v, h] = torch.round(x_rec_blk_0_255) / 255.0 - 0.5
                        zhat[:, :, v, h] = x_rec_blk.clamp_(-0.5, 0.5)
                # calculate loss
                xhat = zhat
                if bool_postprocess:  # apply postprocessing module ???
                    xhat = self.postpm(xhat)
                    xhat.clamp_(-0.5, 0.5)
                rd_loss, mse_loss, rate_loss = self.valid_loss(x, xhat, self_infos)
                self.rcrec_logger(rd_loss.item(), mse_loss.item(), rate_loss.item())
                # for this img : append mse, get psnr, print rdloss etc
                mse_losses.append(mse_loss.item())
                psnr = 10.0 * torch.log10(1.0 / mse_loss.clone().detach())
                title_txt = 'RDLoss:{:.3f} MSE/PSNR:{:.5f}/{:.2f} Rate:{:.3f}'.format(rd_loss.item(), mse_loss.item(),
                                                                                         psnr.item(), rate_loss.item())
                print("Image {:2d} --> ".format(batch_idx) + title_txt)
                # Plot reconstructed images ?
                if False: # True:
                    display_image_in_actual_size(x, self.block_size, self.device) # original img
                    if bool_postprocess:
                        display_image_in_actual_size(zhat, self.block_size, self.device, title_text=title_txt)
                    display_image_in_actual_size(xhat, self.block_size, self.device, title_text=title_txt)  # reconstructed img
                # Write/save reconstructed image to file under exp_/valid_set directory
                xhat_img = arrange_channel_dim_to_block_pixels(xhat + 0.5, self.block_size, self.device)
                self.data_loader.valid_loader.dataset.save_valid_reco_img(xhat_img[0], batch_idx, self.config.checkpoint_dir)
            valid_rd_loss, valid_mse_loss, valid_rate_loss, _ = self.rcrec_logger.display(lr=0.0, typ='va')
            self.logger.info(f'avg_psnr = {10.0 * torch.log10(1.0 / torch.tensor(mse_losses)).mean().item():.2f}')
            # self.scheduler.step(valid_rd_loss)
            return valid_rd_loss

    @torch.no_grad()
    def update_model(self, force=False):  # prepares entropy model for arithmetic coding/decoding
        self.model0.eval()
        with torch.no_grad():
            self.model0.update(force=force)
            # save model back ? with diff name ?
            fname = self.config.modelbest_file_load + "_updated"
            self.save_checkpoint(filename=fname, is_best=False, acl_itr=None, rr=None)

    @torch.no_grad()
    def eval_model(self, bool_postprocess=False):  # runs model with arithmetic coding and decoding
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        if self.device == torch.device("cpu"):
            torch.set_num_threads(1)
        bool_decode = True  # perform also decoding of bitstream ? Use it to verify encoder decoder match, but increases total run time
        self.model0.eval()
        with torch.no_grad():
            # first update_model because we cannot save and load updated model since we get errors with _offset size ...
            self.update_model(force=True)
            # get receptive field size
            L, R, U = self.get_lru_(self.config.KS, 'compress')
            # list to store each pictures mse individually
            mse_losses, msssim_losses, msssimdb_losses = [], [], []
            # send each image in valiadation/test set to inference
            B = self.block_size
            for batch_idx, (x, zhat) in enumerate(self.data_loader.valid_loader):
                x = x.to(self.device)
                # shift pixels to -0.5, 0.5 range
                x = (x - 0.5) * 1
                # pad img on right and bottom to make H,W mulitple of self.block_size
                h, w = x.size(2), x.size(3)
                new_h, new_w = (h + B - 1) // B * B, (w + B - 1) // B * B
                padding_bottom, padding_right = new_h - h, new_w - w
                x_padded = F.pad(x, (0, padding_right, 0, padding_bottom), mode='replicate')  # mode="constant",value=0)
                # arrange pixels inside a block to channel dimension (Note : make sure patch_Size is multiple of B)
                if self.block_size > 1:
                    x_padded = arrange_block_pixels_to_channel_dim(x_padded, self.block_size, self.device)
                # compress img
                start = time.time()
                bitstream, xhat_enc = self.model0.compress(x_padded, [L, R, U], self.config.M)
                enc_time = time.time() - start
                xhat_enc = F.pad(xhat_enc, (0, -padding_right, 0, -padding_bottom))
                # decompress bitstream
                if bool_decode:
                    start = time.time()
                    xhat_dec = self.model0.decompress(bitstream, [L, R, U], x_padded.shape, self.config.M, x_padded.device)
                    dec_time = time.time() - start
                    # check reconstructions from encoder and decoder
                    xhat_dif = torch.abs(xhat_enc - xhat_dec)
                    xhat_dif_mad, xhat_dif_max, xhat_dif_min = xhat_dif.mean(), xhat_dif.max(), xhat_dif.min()
                # apply postprocessing module ???
                if bool_postprocess:
                    xhat_enc = self.postpm(xhat_enc)
                    xhat_enc.clamp_(-0.5, 0.5)
                # calculate loss
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                bpp = len(bitstream) * 8.0 / num_pixels
                xhat_enc_img = arrange_channel_dim_to_block_pixels(xhat_enc, self.block_size, self.device)
                mse = F.mse_loss(x, xhat_enc_img).item()
                # mse = F.mse_loss(x, torch.round((xhat_enc_img + 0.5).mul(255).clamp_(0, 255))/255.0 - 0.5).item()  # quantize reconstruction to 8 bits ? makes psnr results signif. worse.
                if bool_decode:
                    xhat_dec_img = arrange_channel_dim_to_block_pixels(xhat_dec, self.block_size, self.device)
                    mse_dec = F.mse_loss(x, xhat_dec_img).item()
                rd_loss = bpp + self.lambda_ * mse
                psnr = -10 * math.log10(mse)
                msssim = ms_ssim(x + 0.5, xhat_enc_img + 0.5, data_range=1.0).item()
                msssimdb = -10 * math.log10(1.0-msssim)
                # log losses
                self.rcrec_logger(rd_loss, mse, bpp)
                # Plot and save reconstructed images ?
                if False:
                    # display_image_in_actual_size(x, self.block_size, self.device) # original img
                    display_image_in_actual_size(xhat_enc, self.block_size, self.device, title_text=title_txt)  # reconstructed img
                    display_image_in_actual_size(xhat_dec, self.block_size, self.device, title_text=title_txt)
                # Write/save reconstructed image to file under exp_/valid_set directory
                name_orig_file = self.data_loader.valid_loader.dataset.save_valid_reco_img(xhat_enc_img[0] + 0.5, batch_idx, self.config.checkpoint_dir)
                # for this img : append mse, get psnr, print rdloss etc
                mse_losses.append(mse), msssim_losses.append(msssim), msssimdb_losses.append(msssimdb)
                if not bool_decode:
                    dec_time, xhat_dif_mad, xhat_dif_max, xhat_dif_min = 0.0, -1.0, -1.0, -1.0
                title_txt = 'RDLoss:{:.3f} MSE/PSNR:{:.5f}/{:.2f} Rate:{:.3f} MS-SSIM/dB:{:.6f}/{:.2f} Enc/DecTime:{:.1f}/{:.1f} ' \
                            'Enc-Dec.Mad/Max/Min:{:.2f}/{:.2f}/{:.2f} ({})'.format(rd_loss, mse, psnr, bpp, msssim, msssimdb,
                                                                              enc_time, dec_time,
                                                                              xhat_dif_mad*255, xhat_dif_max*255, xhat_dif_min*255, name_orig_file)
                self.logger.info("Image {:2d} --> ".format(batch_idx) + title_txt)
            valid_rd_loss, valid_mse_loss, valid_rate_loss, _ = self.rcrec_logger.display(lr=0.0, typ='va')
            self.logger.info(f'avg_psnr = {10.0 * torch.log10(1.0 / torch.tensor(mse_losses)).mean().item():.2f}  '
                             f'avg_msssim = {torch.tensor(msssim_losses).mean().item():.8f} '
                             f'avg_msssimdb = {torch.tensor(msssimdb_losses).mean().item():.2f}')

    @torch.no_grad()
    def generate_training_set_next_acl_itr(self, dt_loader=None):
        if dt_loader is None:
            dt_loader = self.data_loader.train_loader
        self.model0.eval()
        # self.model1.eval()
        # self.model2.eval()
        with torch.no_grad():
            self.logger.info('Attempting to generate new training set for next ACL iteration ...')
            mse_losses = []
            batch_idx_str = " "
            for batch_idx, (x, zhat) in enumerate(dt_loader):
                # get the original images/data
                x = x.to(self.device)
                x = (x - 0.5) * 1
                # get the reconstructed/noisy version (i.e. zhat) of images/data
                zhat = zhat.to(self.device)
                zhat = (zhat - 0.5) * 1
                # arrange pixels inside a block to channel dimension (Note : make sure patch_Size is multiple of B)
                if self.block_size > 1:
                    x = arrange_block_pixels_to_channel_dim(x, self.block_size, self.device)
                    zhat = arrange_block_pixels_to_channel_dim(zhat, self.block_size, self.device)
                # run through model, calculate loss
                xhat0, self_infos0 = self.model0(zhat, x)
                # xhat1, self_infos1 = self.model1(xhat0, x)
                # xhat2, self_infos2 = self.model2(xhat1, x)
                # !!! clip xhat to [-0.5 to 0.5], this might give better mse (plotting it gives warnings for clipping)
                xhat0.clamp_(-0.5, 0.5)  # this in place version of xhat = torch.clamp(xhat, -0.5, 0.5)
                rd_loss, mse_loss, rate_loss = self.valid_loss(x, xhat0, self_infos0)
                self.gents_logger(rd_loss.item(), mse_loss.item(), rate_loss.item())
                mse_losses.append(mse_loss.item())
                # Write reconstructed image to new training set (trainset_i) folder for next ACL iteration
                xhat_img = arrange_channel_dim_to_block_pixels(xhat0 + 0.5, self.block_size, self.device)
                dt_loader.dataset.save_img(xhat_img[0], batch_idx)
                # write info about how many images saved/generated
                if batch_idx % 100 == 0:
                    batch_idx_str = batch_idx_str + "{:5d} .. ".format(batch_idx)
                    if batch_idx % 1000 == 0:
                        self.logger.info(batch_idx_str)
                        batch_idx_str = " "
            valid_rd_loss, valid_mse_loss, valid_rate_loss, _ = self.gents_logger.display(lr=0.0, typ='va')
            self.logger.info(f'avg_psnr = {10.0 * torch.log10(1.0 / torch.tensor(mse_losses)).mean().item():.2f}')

    @torch.no_grad()
    def generate_training_set_postproc_mdl(self):
        self.model0.eval()
        with torch.no_grad():
            # get receptive field size
            L, R, U = self.get_lru_(self.config.KS)
            # list to store each pictures mse individually
            self.logger.info('Attempting to generate new training set with recu recos for training postprocessing network ...')
            mse_losses = []
            batch_idx_str = " "
            for batch_idx, (x, _) in enumerate(self.data_loader.train_loader):
                x = x.to(self.device)
                # shift pixels to -0.5, 0.5 range
                x = (x - 0.5) * 1
                # arrange pixels inside a block to channel dimension (Note : make sure patch_Size is multiple of B)
                if self.block_size > 1:
                    x = arrange_block_pixels_to_channel_dim(x, self.block_size, self.device)
                # get inputs shape
                bt, ch, hg, wd = x.shape
                # get a reconstructed version of data which is initially all zeros
                zhat = torch.zeros_like(x)
                self_infos = torch.zeros(bt, self.config.M, hg, wd, device=x.device)  # note the channels
                # run through model, calculate loss, update one reconstruction pixel in zhat, do again...
                for v in range(0, hg):
                    # print('{:d} '.format(v), end='')  # print which line is being coded
                    for h in range(0, wd):
                        LL = max(0, h - L)
                        RR = min(wd, h + (R+1))
                        UU = max(0, v - U)
                        xhat_tmp, self_infos_tmp = self.model0(zhat[:, :, UU:v+1, LL:RR], x[:, :, UU:v+1, LL:RR])
                        self_infos[:, :, v, h] = self_infos_tmp[:, :, v-UU, h-LL]
                        x_rec_blk = xhat_tmp[:, :, v-UU, h-LL]
                        # x_rec_blk_0_255 = (x_rec_blk + 0.5).mul(255).clamp_(0, 255)
                        # zhat[:, :, v, h] = torch.round(x_rec_blk_0_255) / 255.0 - 0.5
                        zhat[:, :, v, h] = x_rec_blk.clamp_(-0.5, 0.5)
                # calculate loss
                xhat = zhat
                rd_loss, mse_loss, rate_loss = self.valid_loss(x, xhat, self_infos)
                self.rcrec_logger(rd_loss.item(), mse_loss.item(), rate_loss.item())
                mse_losses.append(mse_loss.item())
                # Write reconstructed image to new training set (trainset_i) folder for next ACL iteration
                xhat_img = arrange_channel_dim_to_block_pixels(xhat + 0.5, self.block_size, self.device)
                self.data_loader.train_loader.dataset.save_img_postproc(xhat_img[0], batch_idx, self.lambda_)
                # write info about how many images saved/generated
                if batch_idx % 100 == 0:
                    batch_idx_str = batch_idx_str + "{:5d} .. ".format(batch_idx)
                    if batch_idx % 500 == 0:
                        self.logger.info(batch_idx_str)
                        batch_idx_str = " "
            valid_rd_loss, valid_mse_loss, valid_rate_loss, _ = self.rcrec_logger.display(lr=0.0, typ='va')
            self.logger.info(f'avg_psnr = {10.0 * torch.log10(1.0 / torch.tensor(mse_losses)).mean().item():.2f}')

    # test should be modified to have actual entorpy coding....
    @torch.no_grad()
    def test(self):
        pass

    # def model_size_estimation(self):
    #     # NOTE: comment/uncomment desired model. If masked convolutions are used, need to subtract out unused parameters

    #     model = self.model0

    #     # from compressai.zoo import mbt2018 as cai_mbt2018
    #     # model = cai_mbt2018(1, metric='mse', pretrained=False, progress=True)
    #     # model = cai_mbt2018(8, metric='mse', pretrained=False, progress=True)

    #     print('---------------Printing paramters--------------------------')
    #     param_size = 0
    #     for name, param in model.named_parameters():
    #         print(name, type(param), param.size())
    #         param_size += param.nelement() * param.element_size()
    #     print('---------------Printing buffers--------------------------')
    #     buffer_size = 0
    #     for name, buffer in model.named_buffers():
    #         print(name, type(buffer), buffer.size())
    #         buffer_size += buffer.nelement() * buffer.element_size()
    #     param_size_mb = param_size / 1024 ** 2
    #     buffer_size_mb = buffer_size / 1024 ** 2
    #     size_all_mb = (param_size + buffer_size) / 1024 ** 2
    #     print('------------------TOT-----------------------')
    #     print(' model param+buffer=total size: {:.2f}+{:.2f}={:.2f}MB'.format(param_size_mb, buffer_size_mb, size_all_mb))
    #     print('------------------END-----------------------')

    # def model_size_estimation(self): 
    #     from torchsummary import summary
    #     #  self.model0 = self.model0.to(self.device) # already done in init
    #     B = self.block_size
    #     summary(self.model0, [(3*B*B, 512//B, 512//B), (3*B*B, 512//B, 512//B)])

    def model_size_estimation(self):
        # NOTE: comment/uncomment desired model. If masked convolutions are used, need to subtract out unused parameters

        model = self.model0
        # from compressai.zoo import mbt2018 as cai_mbt2018
        # model = cai_mbt2018(1, metric='mse', pretrained=False, progress=True)
        # model = cai_mbt2018(8, metric='mse', pretrained=False, progress=True)

        print('---------------Printing/calculating buffers--------------------------')
        buffer_num_elements = 0
        buffers_dic = {}
        for name, buffer in model.named_buffers():
            print(name, '\t',  type(buffer), '\t', buffer.size(), '\t', buffer.nelement())
            buffer_num_elements += buffer.nelement()
            buffers_dic[name] = buffer.size()
                # Calculate totals
        print('------------------BUFFERS--------------------')
        print(' number of model BUFFERS: {:d} '.format(buffer_num_elements))
        print('------------------END-----------------------')
        print('---------------Printing/calculating parameters--------------------------')
        param_num_elements = 0
        param_inactive_num_elements = 0
        for name, param in model.named_parameters():
            print(name, '\t',  type(param), '\t', param.size(), '\t', param.nelement())
            param_num_elements += param.nelement()
            # Check if param is weight and if so if there is a mask associated with this param, and num of inactive weigts
            if name.rsplit('.', 1)[1] == 'weight':
                param_shape = param.shape
                if len(param_shape) == 4 and param_shape[-1] > 1 or param_shape[-2] > 1: # conv and kernel size larger than 1x1
                    base_string = name.rsplit('.', 1)[0]
                    mask_key = base_string + '.mask'
                    if mask_key in buffers_dic:
                        mask_shape = buffers_dic[mask_key] 
                        weights_inactive = (mask_shape[-1] * mask_shape[-2]) // 2
                        weights_ratio = weights_inactive / (mask_shape[-1] * mask_shape[-2])
                        param_nelement_inactive = int(param.nelement() * weights_ratio)
                        print(name, '\t',  type(param), '\t', param.size(), '\t', param_nelement_inactive, ' *** this is the num of INactive weights due to mask in masked conv')
                        param_inactive_num_elements += param_nelement_inactive
        # Calculate totals
        print('------------------TOT-----------------------')
        print(' number of model all PARAMS:\t {:d} '.format(param_num_elements))
        print(' number of model INACTIVE PARAMS:\t {:d} (due to mask in masked convolution)'.format(param_inactive_num_elements))
        print(' number of model active PARAMS:\t {:d} '.format(param_num_elements - param_inactive_num_elements))
        print('------------------END-----------------------')


    def flops_estimation(self):
        def prepare_input(resolution):
            x = torch.Tensor(1, *resolution)
            zhat = torch.Tensor(1, *resolution)
            x = x.to(self.device)
            zhat = zhat.to(self.device)
            return dict(x = x, zhat = zhat)
        from ptflops import get_model_complexity_info
        B = self.block_size
        net = self.model0
        macs, params = get_model_complexity_info(net, input_res=(3*B*B, 512//B, 512//B), input_constructor=prepare_input,
                                                    as_strings=True, backend='aten', #'aten', #'pytorch', 
                                                    print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    # def flops_estimation(self):
    #     from ptflops import get_model_complexity_info
        
    #     from compressai.zoo import mbt2018 as cai_mbt2018
    #     # model = cai_mbt2018(1, metric='mse', pretrained=False, progress=True)
    #     model = cai_mbt2018(8, metric='mse', pretrained=False, progress=True)
    #     B = 1
    #     net = model
    #     macs, params = get_model_complexity_info(net, input_res=(3*B*B, 512//B, 512//B),
    #                                                 as_strings=True, backend='pytorch', #'aten', #'pytorch', 
    #                                                 print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def arrange_block_pixels_to_channel_dim(x, B, dev):
    C, H, W = x.shape[1], x.shape[2], x.shape[3]
    y = torch.empty(x.shape[0], C*(B**2), H//B, W//B, device=dev)
    for v in range(0, B, 1):
        for h in range(0, B, 1):
            indd = (v*B+h)*C
            y[:, indd:indd+C, :, :] = x[:, :, v::B, h::B]
    return y


def arrange_channel_dim_to_block_pixels(y, B, dev):
    C, H, W = y.shape[1], y.shape[2], y.shape[3]
    C = C//(B**2)
    H = H*B
    W = W*B
    x = torch.empty(y.shape[0], C, H, W, device=dev)
    for v in range(0, B, 1):
        for h in range(0, B, 1):
            indd = (v*B+h)*C
            x[:, :, v::B, h::B] = y[:, indd:indd+C, :, :]
    return x
