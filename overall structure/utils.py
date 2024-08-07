import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import os
import re


def draw_recon(x, x_recon):
    x_l, x_recon_l = x.tolist(), x_recon.tolist()
    result = [None] * (len(x_l) + len(x_recon_l))
    result[::2] = x_l
    result[1::2] = x_recon_l
    return torch.FloatTensor(result)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


def make_dataloader(args):

    test_loader = None
    train_loader = None
    llm_file = "/root/DEARwithtc/tmpfinal.txt"
    llmarr = []
    if 'pendulum' in args.dataset:
        label_idx = range(4)
    else:
        if args.labels == 'smile':
            label_idx = [31, 20, 19, 21, 23, 13]
        elif args.labels == 'age':
            # label_idx = [39, 20, 28, 18, 13, 3]
            label_idx = [5, 4, 20, 24, 9, 18]
            tmplabel_idx = [16, 10, 2, 8, 12, 4]
            # tmplabel_idx = [18, 19, 20, 21, 22, 23]
            with open(llm_file, "r") as llmfile:
                llminfo = llmfile.readlines()
                for line in llminfo:
                    line = re.sub('[^\d]', ' ', line)
                    process = list(map(int,line.split()))
                    if len(process) < 25:
                        for _ in range(25):
                            process.append(0)
                    process = np.array(process)
                    # print(process)
                    process = torch.tensor(process[tmplabel_idx])
                    process = process/5
                    # print(process)
                    # return
                    # print(process.shape)
                    llmarr.append(process)

                # print(llmarr)
                llmarr = torch.stack(llmarr)
                # print(llmarr.shape)


    if args.dataset == 'celeba':

        trans_f = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CelebA(args.data_dir, split='train', download=False, transform=trans_f)
        # print(train_set.attr[:, label_idx])
        # print(train_set.attr[:, label_idx].shape)

        # with open("check.txt", "w") as file:
        #     train_set.attr = train_set.attr.float()
        #     print(train_set.attr)
        #     file.write(str(train_set.attr.numpy()))
        #     file.write('\n')
        #     train_set.attr[:1000, label_idx] = llmarr
        #     print(train_set.attr)
        #     file.write(str(train_set.attr.numpy()))
        #     file.write('\n')
        # train_set.attr[:, label_idx] = train_set.attr[:, label_idx] / 2
        # print(train_set.attr[:, label_idx])
        # print(train_set.attr[:, label_idx].shape)
        train_set.attr = train_set.attr.float()

        train_set.attr[:, label_idx] = llmarr

        indices = range(args.datasetnum)
        train_set = torch.utils.data.Subset(train_set, indices)
        # train_set = train_set[:(len(train_set)//2)]

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                                   drop_last=True, num_workers=4)

    elif 'pendulum' in args.dataset:
        train_set = dataload_withlabel(args.data_dir, image_size = args.image_size,
                                       mode='train', sup_prop=args.sup_prop)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    return train_loader, test_loader



def check_for_CUDA(sagan_obj):
    if not sagan_obj.config.disable_cuda and torch.cuda.is_available():
        print("CUDA is available!")
        sagan_obj.device = torch.device('cuda')
        sagan_obj.config.dataloader_args['pin_memory'] = True
    else:
        print("Cuda is NOT available, running on CPU.")
        sagan_obj.device = torch.device('cpu')

    if torch.cuda.is_available() and sagan_obj.config.disable_cuda:
        print("WARNING: You have a CUDA device, so you should probably run without --disable_cuda")


class dataload_withlabel(torch.utils.data.Dataset):
    def __init__(self, root, label_file=None, image_size=64, mode="train", sup_prop=1., num_sample=0):
        # label_file: 'pendulum_label_downstream.txt'

        self.label_file = label_file
        if label_file is not None:
            self.attrs_df = pd.read_csv(os.path.join(root, label_file))
            # attr = self.attrs_df[:, [1,2,3,7,5]]
            self.split_df = pd.read_csv(os.path.join(root, label_file))
            splits = self.split_df['partition'].values
            split_map = {
                "train": 0,
                "valid": 1,
                "test": 2,
                "all": None,
            }
            split = split_map[verify_str_arg(mode.lower(), "split",
                                             ("train", "valid", "test", "all"))]
            mask = slice(None) if split is None else (splits == split)
            self.mask = mask
            np.random.seed(2)
            if num_sample > 0:
                idxs = [i for i, x in enumerate(mask) if x]
                not_sample = np.random.permutation(idxs)[num_sample:]
                mask[not_sample] = False
            self.attrs_df = self.attrs_df.values
            self.attrs_df[self.attrs_df == -1] = 0
            self.attrs_df = self.attrs_df[mask][:, [0,1,2,3,6]]
            self.imglabel = torch.as_tensor(self.attrs_df.astype(np.float))
            self.imgs = []
            for i in range(3):
                mode1 = list(split_map.keys())[i]
                root1 = root + mode1
                imgs = os.listdir(root1)
                self.imgs += [os.path.join(root, mode1, k) for k in imgs]
            self.imgs = np.array(self.imgs)[mask]
        else:
            root = root + mode
            imgs = os.listdir(root)
            self.imgs = [os.path.join(root, k) for k in imgs]
            self.imglabel = [list(map(float, k[:-4].split("_")[1:])) for k in imgs]
        self.transforms = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.ToTensor()])
        np.random.seed(2)
        self.n = len(self.imgs)
        self.available_label_index = np.random.choice(self.n, int(self.n * sup_prop), replace=0)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        if not (idx in self.available_label_index):
            label = torch.zeros(4).long() - 1
        else:
            if self.label_file is None:
                label = torch.from_numpy(np.asarray(self.imglabel[idx]))
            else:
                label = self.imglabel[idx]
        pil_img = Image.open(img_path).convert('RGB')
        array = np.array(pil_img)
        array1 = np.array(label)
        label = torch.from_numpy(array1)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,3)
            data = torch.from_numpy(pil_img)
        return data, label.float()

    def __len__(self):
        return len(self.imgs)
