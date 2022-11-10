import os
import argparse
import logging
from statistics import mode

from handlers.trainer import Trainer
from utils.general import save_script_args

# Load logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Model arguments
    model_parser = argparse.ArgumentParser(description='Arguments for system and model configuration')
    model_parser.add_argument('--path', type=str, help='path to experiment')
    model_parser.add_argument('--transformer', default='electra-base',type=str, help='[bert, roberta, electra ...]')
    model_parser.add_argument('--maxlen', default=512, type=int, help='max length of transformer inputs')
    model_parser.add_argument('--num-classes', default=2, type=int, help='number of classes (3 for NLI)')

    ### Training arguments
    train_parser = argparse.ArgumentParser(description='Arguments for training the system')
    train_parser.add_argument('--dataset', default='imdb', type=str, help='dataset to train the system on')
    train_parser.add_argument('--lim', default=None, type=int, help='size of data subset to use for debugging')
    
    train_parser.add_argument('--epochs', default=10, type=int, help='size of data subset to use for debugging')
    train_parser.add_argument('--bsz', default=4, type=int, help='size of data subset to use for debugging')
    train_parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    
    train_parser.add_argument('--log-every', default=200, type=int, help='logging training metrics every number of examples')
    train_parser.add_argument('--wandb', action='store_true', help='if set, will log to wandb')
    train_parser.add_argument('--device', default='cuda', type=str, help='selecting device to use')

    # Parse system input arguments
    model_args, moargs = model_parser.parse_known_args()
    train_args, toargs = train_parser.parse_known_args()

    # Making sure no unkown arguments are given
    assert set(moargs).isdisjoint(toargs), f"{set(moargs) & set(toargs)}"

    logger.info(model_args.__dict__)
    logger.info(train_args.__dict__)
    
    trainer = Trainer(model_args.path, model_args)
    trainer.train(train_args)