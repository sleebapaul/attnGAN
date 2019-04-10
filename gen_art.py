from __future__ import print_function

# From libraries
import argparse
import datetime
import os
import pprint
import random
import sys
import time

import dateutil.tz
import numpy as np

import torch
import torchvision.transforms as transforms
from nltk.tokenize import RegexpTokenizer

# From files
from datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from trainer import condGANTrainer as trainer

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--input_text',
                        help='Input text to convert into a image using AttnGAN',
                        default='Mary had a little lamb', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfg/eval_coco.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--model_path', dest='model_path',
                        type=str, default='')
    parser.add_argument('--textencoder_path',
                        dest='textencoder_path', type=str, default='')
    parser.add_argument('--output_dir', dest='output_dir',
                        type=str, default='output')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example_from_text(input_text, output_dir, wordtoix, algo):
    '''
    Generate image from example sentence
    '''

    # a list of indices for a sentence
    captions = []
    cap_lens = []
    data_dic = {}
    if len(input_text) == 0:
        return 0
    sent = input_text.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent.lower())
    if len(tokens) == 0:
        print('No tokens for: ', sent)
        return 0

    rev = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            rev.append(wordtoix[t])
    captions.append(rev)
    cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    data_dic["data"] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(output_dir, data_dic)


if __name__ == "__main__":
    import sys
    print(sys.version)

    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    if args.model_path != '':
        cfg.TRAIN.NET_G = args.model_path

    if args.textencoder_path != '':
        cfg.TRAIN.NET_E = args.textencoder_path

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = './output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''
        Generate images from pre-extracted embeddings
        '''
        if cfg.B_VALIDATION:
            # generate images for the whole valid dataset
            algo.sampling(split_dir)
        else:
            # generate images for customized captions
            gen_example_from_text(
                args.input_text, args.output_dir, dataset.wordtoix, algo)
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
