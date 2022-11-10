import os
import json

import argparse
import logging
from statistics import mode

from handlers.evaluater import Evaluator
from utils.general import save_script_args

# Load logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    ### Decoding arguments
    eval_parser = argparse.ArgumentParser(description='Arguments for training the system')
    eval_parser.add_argument('--path', type=str, help='path to experiment')
    eval_parser.add_argument('--dataset', type=str, help='dataset to train the system on')
    eval_parser.add_argument('--mode', default='test', type=str, help='which data split to evaluate on')
    eval_parser.add_argument('--device', default='cuda', type=str, help='selecting device to use')

    args = eval_parser.parse_args()

    print(args)
    
    evaluator = Evaluator(args.path, args.device)
    preds = evaluator.load_preds(args.dataset, args.mode)
    labels = evaluator.load_labels(args.dataset, args.mode)
    acc = evaluator.calc_acc(preds, labels)

    print(f'accuracy: {acc:.2f}')