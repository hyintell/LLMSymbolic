import time
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--jar_path", type=str)
    #parser.add_argument("--task_num", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--lm_path", default="lm_model")
    parser.add_argument("--simplification_str", default="easy")
    parser.add_argument("--beams", type=int, default=16)
    #parser.add_argument("--max_episode_per_file", type=int, default=1000)
    parser.add_argument("--mode", default="bc")
    parser.add_argument("--set", default="dev")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--model_parallelism_size", type=int, default=1)    # the number of GPUs.

    parser.add_argument('--historySavePrefix', default='t5saveout', type=str)
    parser.add_argument('--maxHistoriesPerFile', default=1000, type=int)

    # Number of variations to run
    parser.add_argument("--num_variations", type=int, default=100)

    # TextWorldExpress
    parser.add_argument("--jar_path", type=str,
                        help="Path to the TextWorldExpress jar file. Default: use builtin.")
    parser.add_argument("--game_name", type=str, choices=['arithmetic', 'twc', 'mapreader', 'sorting', 'twc-easy'], default='arithmetic',
                        help="Specify the game to play. Default: %(default)s")
    parser.add_argument("--game_params", type=str, default='',
                        help="TODO: This currently is not supported")



    # Mode select: Training data generation, OR running the model.
    parser.add_argument("--train_or_eval", type=str, choices=['train-gen', 'eval'], default='eval',
                        help="Specify whether to generate training data, evaluate the model. Default: %(default)s")

    parser.add_argument('--traindataSavePrefix', default='t5goldout', type=str)
    parser.add_argument('--useSymbolicModules', default='', type=str)

    

    # ================================= Add by Reed ================================= #

    # Model
    parser.add_argument('--agent', default='', type=str, help='agent to play the game', choices=['LLM', 'T5'])
    parser.add_argument('--fewshot', default=False, action='store_true', help='few-shot learning')

    # Data Collection
    parser.add_argument('--save-data', default=False, action='store_true', help='collect data')
    parser.add_argument('--data-save-path', default='data', type=str, help='path to save data')
    
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')


    parser.add_argument('--with-wandb', default=False, action='store_true', help='Enables Weights and Biases')
    parser.add_argument('--wandb-entity', default='reedzyd', type=str, help='WandB username (entity).')
    parser.add_argument('--wandb-project', default='SymbolicGameGuidedbyLLM', type=str, help='WandB "Project"')
    parser.add_argument('--wandb-group', default=None, type=str, help='WandB "Group". Name of the env by default.')
    parser.add_argument('--wandb-job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb-tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb-key', default=None, type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb-dir', default=None, type=str, help='the place to save WandB files')
    parser.add_argument('--wandb-experiment', default='', type=str, help='Identifier to specify the experiment')
    # =============================================================================== #

    args = parser.parse_args()

    args.timestamp = time.strftime('%Y-%m-%d-%H_%M_%S' + '_' + str(np.random.randint(100)))


    # Post-processing
    args.game_params = ""

    if (args.game_name == "twc-easy"):
        args.game_name = "twc"
        paramStr = "numLocations=1,numItemsToPutAway=1,includeDoors=0,limitInventorySize=0"     # Equivalent of TWC-Easy
        args.game_params = paramStr


    return args
