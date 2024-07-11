import os
import sys
import torch
import logging
import numpy as np
from PIL import Image
from PIL import ImageOps
        
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop, ToTensor, Compose, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter
from torchvision.utils import save_image
import torchvision.transforms.functional as TTF


class ImageDataLoader_ACL():
    def __init__(self, config):
        train_datas = [config.train_data_1, config.train_data_2, config.train_data_3, config.train_data_4]
        train_datas = train_datas[0:config.num_train_dirs]
        train_patch_size = 0 if config.mode == "gen_train_set" else config.patch_size
        # if config.mode == "gen_train_set" or config.mode == "gen_train_set_postproc" or "train_post_proc_mdl":
        if any(config.mode == cfgmd for cfgmd in ["gen_train_set", "gen_train_set_postproc", "train_postproc_mdl"]):
            self.train_dataset = ImageDataset_ACL(train_datas,
                                              train_patch_size,
                                              acl_bool=config.acl_bool,
                                              acl_itr=config.acl_itr,
                                              augment=False,
                                              zhat_fldr_ext=('__recurecos_' + str(config.lambda_) if config.mode == "train_postproc_mdl" else None),
                                              session=config.session)
        else:
            self.train_dataset = ImageDataset_ACL(train_datas,
                                              train_patch_size,
                                              acl_bool=config.acl_bool,
                                              acl_itr=config.acl_itr,
                                              augment=True,
                                              session=config.session)
        self.valid_dataset = ImageDataset_ACL(config.valid_data,
                                          config.val_patch_size,
                                          acl_bool=config.acl_bool,
                                          acl_itr=(0 if config.mode == "train_postproc_mdl" else config.acl_itr),
                                          augment=False,
                                          session=config.session)

        num_workers = 0 if config.mode == 'debug' else 4  # 0 # 4

        # if config.mode == "gen_train_set" or config.mode == "gen_train_set_postproc":
        if any(config.mode == cfgmd for cfgmd in ["gen_train_set", "gen_train_set_postproc"]):
            self.train_loader = DataLoader(self.train_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=False,
                                            drop_last=False)
        else:  # "train_post_proc_mdl" should be here not above since we train, i.e. want batch_size>1 etc.
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           drop_last=False)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=config.val_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=False,
                                       drop_last=False)


class ImageDataset_ACL(Dataset):
    def __init__(self, root, size, acl_bool=False, acl_itr=0, augment=False, zhat_fldr_ext=None, session='A'):
        self.size = size  # fatih added this to know size in getitem to resize there wif image is smaller than size
        self.acl_bool = acl_bool
        self.acl_itr = acl_itr
        self.augment = augment
        self.augment_p = 0.5
        self.zhat_fldr_ext = zhat_fldr_ext
        self.ss = session + "_"
        # self.colorjitter = ColorJitter(hue=0.07)
        try:
            if isinstance(root, str):
                self.image_files = [os.path.join(root, f)  for f in os.listdir(root) if (f.endswith('.png') or f.endswith('.jpg'))]
                # if training data for ACL will be generated
                if self.acl_bool and self.acl_itr > 0:
                    if self.zhat_fldr_ext is None:
                        root_acl = root + '__acl_' + self.ss + str(self.acl_itr)
                    else:
                        root_acl = root + self.zhat_fldr_ext
                    if not os.path.exists(root_acl):
                        print('ACL zhat training_or_validation set directory ' + root_acl + ' could not be found. Exiting.'), sys.exit(1)
                    self.acl_image_files = [os.path.join(root_acl, f)  for f in os.listdir(root_acl) if (f.endswith('.png') or f.endswith('.jpg'))]
                    assert len(self.acl_image_files) == len(self.image_files)
                elif self.acl_bool and self.acl_itr == 0:
                    self.acl_image_files = self.image_files
            else:
                self.image_files = []
                for i in range(0, len(root)):
                    self.image_files_temp = [os.path.join(root[i], f)  for f in os.listdir(root[i]) if (f.endswith('.png') or f.endswith('.jpg'))]
                    self.image_files = self.image_files + self.image_files_temp
                # if training data for ACL will be generated
                if self.acl_bool and self.acl_itr > 0:
                    self.acl_image_files = []
                    for i in range(0, len(root)):
                        if self.zhat_fldr_ext is None:
                            root_acl = root[i] + '__acl_' + self.ss + str(self.acl_itr)
                        else:
                            root_acl = root[i] + self.zhat_fldr_ext
                        if not os.path.exists(root_acl):
                            print('ACL zhat training set directory ' + root_acl + ' could not be found. Exiting.'), sys.exit(1)
                        self.acl_image_files_temp = [os.path.join(root_acl, f)  for f in os.listdir(root_acl) if (f.endswith('.png') or f.endswith('.jpg'))]
                        self.acl_image_files = self.acl_image_files + self.acl_image_files_temp
                    assert len(self.acl_image_files) == len(self.image_files)
                elif self.acl_bool and self.acl_itr == 0:
                    self.acl_image_files = self.image_files
        except:
            logging.getLogger().exception('Dataset could not be found. Drive might be unmounted.', exc_info=False)
            sys.exit(1)
        if size == 0:
            self.transforms = Compose([ToTensor()])
        else:
            self.transforms = Compose([CenterCrop(size), ToTensor()])

    def set_augment(self, bool_aug):
        self.augment = bool_aug

    def set_acl_itr(self, acl_itr):
        self.acl_itr = acl_itr

    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, i):
        # NOTE check if the output is normalized between 0-1
        # img = pil_loader(self.image_files[i])
        # return self.transforms(img)
        img = pil_loader(self.image_files[i])
        # check size of image and resize it if width or height less than requested size.. no more in acl

        if not self.acl_bool:
            if self.augment:
                if torch.rand(1) < self.augment_p:
                    img = TTF.hflip(img)
                if torch.rand(1) < self.augment_p:
                    img = TTF.vflip(img)
            return self.transforms(img)
        else:
            img_zhat = pil_loader(self.acl_image_files[i])
            if self.augment:
                if torch.rand(1) < self.augment_p:
                    img = TTF.hflip(img)
                    img_zhat = TTF.hflip(img_zhat)
                if torch.rand(1) < self.augment_p:
                    img = TTF.vflip(img)
                    img_zhat = TTF.vflip(img_zhat)
                # if torch.rand(1) < self.augment_p:  # jitter only zhat
                #     img_zhat = self.colorjitter(img_zhat)
            return self.transforms(img), self.transforms(img_zhat)

    def save_img(self, img, i):
        lggr = logging.getLogger()
        path_pieces = self.image_files[i].split('/')
        dir_name = '/'.join(path_pieces[0:-1]) + '__acl_' + self.ss + str(self.acl_itr+1)
        if not os.path.exists(dir_name):
            lggr.info('Creating directory to save reconstructed images for next ACL itr (did not exist):' + dir_name)
            try:
                os.mkdir(dir_name)
                lggr.info('Directory created.')
            except:
                lggr.info('Directory could not be created. Exiting.'), sys.exit(1)
        elif i == 0:
            lggr.info('Directory to save reconstructed images for next ACL itr already exists (may overwrite files in it): ' + dir_name )
        img_file_name = dir_name + '/' + path_pieces[-1]
        save_image(img, img_file_name)

    def save_img_postproc(self, img, i, lambd):
        lggr = logging.getLogger()
        path_pieces = self.image_files[i].split('/')
        dir_name = '/'.join(path_pieces[0:-1]) + '__recurecos_' + str(lambd)
        if not os.path.exists(dir_name):
            lggr.info('Creating directory to save reconstructed images with recursive reconstruction for training postprocessing module (did not exist): ' + dir_name)
            try:
                os.mkdir(dir_name)
                lggr.info('Directory created.')
            except:
                lggr.info('Directory could not be created. Exiting.'), sys.exit(1)
        elif i == 0:
            lggr.info('Directory to save reconstructed images with recursive reconstruction for training postprocessing module already exists (may overwrite files in it): ' + dir_name )
        img_file_name = dir_name + '/' + path_pieces[-1]
        save_image(img, img_file_name)

    def save_valid_reco_img(self, img, i, checkpoint_dir):
        lggr = logging.getLogger()
        path_pieces_orig_file = self.image_files[i].split('/')
        name_orig_file = path_pieces_orig_file[-1]
        dir_name = checkpoint_dir + '/../' + path_pieces_orig_file[-2]
        if not os.path.exists(dir_name):
            lggr.info('Creating directory to save validation reconstructed images (did not exist):' + dir_name)
            try:
                os.mkdir(dir_name)
                lggr.info('Directory created.')
            except:
                lggr.info('Directory could not be created. Exiting.'), sys.exit(1)
        elif i == 0:
            lggr.info('Directory to save validation reconstructed images already exists (may overwrite files in it): ' + dir_name )
        img_file_name = dir_name + '/' + name_orig_file
        save_image(img, img_file_name)
        return name_orig_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning 
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
