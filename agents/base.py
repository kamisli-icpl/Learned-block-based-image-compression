import logging
import sys
import shutil

import torch
from torch.backends import cudnn


cudnn.benchmark = True
cudnn.enabled = True


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.config_org = config
        self.logger = logging.getLogger("Agent")
        self.best_valid_loss = float('inf')  # 0
        self.best_validrr_loss = float('inf')  # 0
        self.prev_aclitr_best_valid_loss = float('inf')
        self.prev_aclitr_best_validrr_loss = float('inf')
        self.current_epoch = 0
        self.current_iteration = 0
        # !! original version which only supports cuda but not cpu
        # self.device = torch.device("cuda")
        # self.cuda = torch.cuda.is_available() & self.config.cuda
        # self.manual_seed = self.config.seed
        # torch.cuda.manual_seed(self.manual_seed)
        # torch.cuda.set_device(self.config.gpu_device)
        # !! my modifed version which also supports cpu
        self.manual_seed = self.config.seed
        self.cuda = torch.cuda.is_available() & self.config.cuda
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
        
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation without recursive reconstrcution, where zhat is simply a noisy version of original
        :return:
        """
        raise NotImplementedError

    def validate_recu_reco(self):
        """
        One cycle of model validation with recursive reconstruction, i.e zhat is actual reconstruction
        :return:
        """
        raise NotImplementedError

    def validate_recu_reco_fast(self):
        """
        One cycle of model validation with recursive reconstruction, i.e zhat is actual reconstruction
        :return:
        """
        raise NotImplementedError

    def test(self):
        """
        One cycle of model test
        :return:
        """
        raise NotImplementedError

    def generate_training_set_next_acl_itr(self):
        """
        One cycle of model test
        :return:
        """
        raise NotImplementedError

    def generate_training_set_postproc_mdl(self):
        raise NotImplementedError

    def train_post_proc_mdl(self):
        raise NotImplementedError

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            # checkpoint = torch.load(filename)
            checkpoint = torch.load(filename, map_location=self.device)
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.best_valid_loss = checkpoint['best_valid_loss']  # prev acl iter may have better valid loss, so better valid loss may never be possible in this acl iter !?
            self.best_validrr_loss = checkpoint['best_validrr_loss']
            self.prev_aclitr_best_valid_loss = checkpoint['prev_aclitr_best_valid_loss']
            self.prev_aclitr_best_validrr_loss = checkpoint['prev_aclitr_best_validrr_loss']
            self.model0.load_state_dict(checkpoint['state_dict0'])
            # self.model1.load_state_dict(checkpoint['state_dict'])
            # self.model2.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.train_logger0.load_state_dict(checkpoint['train_logger0'])
            self.train_logger1.load_state_dict(checkpoint['train_logger1'])
            self.trnit_logger0.load_state_dict(checkpoint['trnit_logger0'])
            self.trnit_logger1.load_state_dict(checkpoint['trnit_logger1'])
            self.valid_logger0.load_state_dict(checkpoint['valid_logger0'])
            self.rcrec_logger.load_state_dict(checkpoint['rcrec_logger'])
            self.amp_scaler.load_state_dict(checkpoint['amp_scaler'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})"
                            .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
            # self.model.to(self.device)   # no need here ?
            # Fix the optimizer cuda error
            if self.cuda and self.config.mode != 'eval_model':  # this if statement is added by fatih later but not tested
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

        except OSError as e:
            self.logger.info("!!! No checkpoint exists from '{}'. Continuing with available parameters..."
                             .format(self.config.checkpoint_dir))
            ##self.logger.info("**First time to train**")
            
    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0, acl_itr=None, rr=False):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'best_valid_loss': self.best_valid_loss,
            'best_validrr_loss': self.best_validrr_loss,
            'prev_aclitr_best_valid_loss': self.prev_aclitr_best_valid_loss,
            'prev_aclitr_best_validrr_loss': self.prev_aclitr_best_validrr_loss,
            'state_dict0': self.model0.state_dict(),
            # 'state_dict1': self.model1.state_dict(),
            # 'state_dict2': self.model2.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_logger0': self.train_logger0.state_dict(),
            'train_logger1': self.train_logger1.state_dict(),
            'trnit_logger0' : self.trnit_logger0.state_dict(),
            'trnit_logger1' : self.trnit_logger1.state_dict(),
            'valid_logger0' : self.valid_logger0.state_dict(),
            'rcrec_logger' : self.rcrec_logger.state_dict(),
            'amp_scaler' : self.amp_scaler.state_dict()
        }
        torch.save(state, self.config.checkpoint_dir + filename)
        if is_best:
            if acl_itr is None:
                shutil.copyfile(self.config.checkpoint_dir + filename,
                                self.config.checkpoint_dir + 'model_best.pth.tar')
            else:
                if rr is False:
                    shutil.copyfile(self.config.checkpoint_dir + filename,
                                    self.config.checkpoint_dir + "model_best_acl_" + str(acl_itr) + ".pth.tar")
                else:
                    shutil.copyfile(self.config.checkpoint_dir + filename,
                                    self.config.checkpoint_dir + "model_best_acl_" + str(acl_itr) + "rr.pth.tar")

    def run(self):
        try:
            if self.config.mode == 'test':
                self.test()
            elif self.config.mode == 'validate':
                self.validate()
            elif self.config.mode == 'validate_recu_reco':
                self.validate_recu_reco()
            elif self.config.mode == 'validate_recu_reco_fast':
                self.validate_recu_reco_fast()
            elif self.config.mode == 'gen_train_set':
                # self.generate_training_set_next_acl_itr()
                self.generate_training_set_next_acl_itr(self.data_loader.train_loader)  # training set
                self.generate_training_set_next_acl_itr(self.data_loader.valid_loader)  # validation set
            elif self.config.mode == 'gen_train_set_postproc':
                self.generate_training_set_postproc_mdl()
            elif self.config.mode == 'train_postproc_mdl':
                self.train_postproc_mdl()
            elif self.config.mode == 'train_one_acl':
                self.train_one_acl()
            elif self.config.mode == 'train_all_acl':
                self.train_all_acl()
            elif self.config.mode == 'debug':
                with torch.autograd.detect_anomaly():
                    self.train_one_acl()
            elif self.config.mode == 'update_model':
                self.update_model(force=True)
            elif self.config.mode == 'eval_model':
                self.eval_model()
            elif self.config.mode == 'model_size_estimation':
                self.model_size_estimation()
            elif self.config.mode == "flops_estimation":
                self.flops_estimation()
            else:
                raise NameError("'" + self.config.mode + "'" 
                                + ' is not a valid training mode.' )
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
        except AssertionError as e:
            raise e
        except Exception as e:
            self.save_checkpoint()
            raise e

    def train_one_acl(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            if not (self.current_epoch+1) % self.config.validate_every:
                valid_loss = self.validate()
                is_best = valid_loss < self.best_valid_loss
                if is_best:
                    self.best_valid_loss = valid_loss
                if self.config.acl_bool:
                    self.save_checkpoint(is_best=is_best, acl_itr=self.config.acl_itr, rr=False)
                else:
                    self.save_checkpoint(is_best=is_best)
            if not (self.current_epoch+1) % self.config.validate_recu_reco_every:
                validrr_loss = self.validate_recu_reco()
                is_best = validrr_loss < self.best_validrr_loss
                if is_best:
                    self.best_validrr_loss = validrr_loss
                if self.config.acl_bool:
                    self.save_checkpoint(is_best=is_best, acl_itr=self.config.acl_itr, rr=True)
                else:
                    self.save_checkpoint(is_best=is_best)
            # if not (self.current_epoch+1) % self.config.test_every:
            #     test_loss = self.test()
            self.current_epoch += 1

    def train_all_acl(self):
        raise NotImplementedError

    def finalize(self):
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
