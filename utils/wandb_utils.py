import logging
import time
import wandb
from argparse import Namespace
import argparse
import numpy as np

def wandb_parser_args():
    parser = argparse.ArgumentParser()

    # WandB
    parser.add_argument('--with-wandb', default=False, action='store_true', help='Enables Weights and Biases')
    parser.add_argument('--wandb-entity', default='reedzyd', type=str, help='WandB username (entity).')
    parser.add_argument('--wandb-project', default='SymbolicGameGuidedbyLLM', type=str, help='WandB "Project"')
    parser.add_argument('--wandb-group', default=None, type=str, help='WandB "Group". Name of the env by default.')
    parser.add_argument('--wandb-job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb-tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb-key', default=None, type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb-dir', default=None, type=str, help='the place to save WandB files')
    parser.add_argument('--wandb-experiment', default='', type=str, help='Identifier to specify the experiment')

    args = parser.parse_args()

    args.timestamp = time.strftime('%Y-%m-%d-%H_%M_%S' + '_' + str(np.random.randint(100)))

    return args

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class WandbArgument:
    # Add your custom arguments here
    with_wandb: bool = field(default=False, metadata={'help': 'Enables Weights and Biases'})
    wandb_entity: str = field(default='reedzyd', metadata={'help': 'WandB username (entity).'})
    wandb_project: str = field(default='SymbolicGameGuidedbyLLM', metadata={'help': 'WandB "Project"'})
    wandb_group: Optional[str] = field(default=None, metadata={'help': 'WandB "Group". Name of the env by default.'})
    wandb_job_type: str = field(default='train', metadata={'help': 'WandB job type'})
    wandb_tags: Optional[List[str]] = field(default_factory=list, metadata={'help': 'Tags can help finding experiments'})
    wandb_key: Optional[str] = field(default=None, metadata={'help': 'API key for authorizing WandB'})
    wandb_dir: Optional[str] = field(default=None, metadata={'help': 'the place to save WandB files'})
    wandb_experiment: str = field(default='', metadata={'help': 'Identifier to specify the experiment'})
    
    def __post_init__(self, ):
        self.timestamp = time.strftime('%Y-%m-%d-%H_%M_%S' + '_' + str(np.random.randint(100)))

def retry(times, exceptions):
    """
    Retry Decorator https://stackoverflow.com/a/64030200/1645784
    Retries the wrapped function/method `times` times if the exceptions listed in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param exceptions: Lists of exceptions that trigger a retry attempt
    :type exceptions: Tuple of Exceptions
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(f'Exception thrown when attempting to run {func}, attempt {attempt} out of {times}')
                    time.sleep(min(2**attempt, 10))
                    attempt += 1

            return func(*args, **kwargs)

        return newfn

    return decorator


def init_wandb(args: Namespace):
    """
    Must call initialization of WandB before summary writer is initialized, otherwise sync_tensorboard does not work.
    """
    if not args.with_wandb:
        logging.info('Weights and Biases integration disabled')
        return

    if args.wandb_group is None:
        args.wandb_group = f'{args.game_name}'

    if 'wandb_unique_id' not in args:
        if len(args.wandb_tags) != 0:
            args.wandb_unique_id = f'{args.wandb_tags}_{args.game_name}_{args.agent}_{args.timestamp}'
        else:
            args.wandb_unique_id = f'{args.game_name}_{args.agent}_{args.timestamp}'

    logging.info(
        f'Weights and Biases integration enabled. Project: {args.wandb_project}, user: {args.wandb_entity}, '
        f'group: {args.wandb_group}, unique_id: {args.wandb_unique_id}')

    # Try multiple times, as this occasionally fails
    @retry(3, exceptions=(Exception,))
    def init_wandb_func():
        wandb.init(
            dir=args.wandb_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            id=args.wandb_unique_id,
            name=args.wandb_unique_id,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            tags=args.wandb_tags,
            resume=False,
            settings=wandb.Settings(start_method='fork'),
        reinit=True
        )

    logging.info('Initializing WandB...')
    try:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        init_wandb_func()
    except Exception as exc:
        logging.error(f'Could not initialize WandB! {exc}')

    wandb.config.update(args, allow_val_change=True)


def finish_wandb(cfg):
    if cfg.with_wandb:
        import wandb
        wandb.run.finish()
