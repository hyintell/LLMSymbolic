# Learn to play symbolic games with LLM guidances
This repository is implemented based the repo for the paper ["Large Language Models Are Neurosymbolic Reasoners"](https://arxiv.org/abs/2401.09334) (AAAI 2024).

## Before using the repo

**1. Install Dependencies:**
```bash
conda create --name t5-neurosymbolic python=3.9
conda activate t5-neurosymbolic
pip install -r requirements.txt
```

**2. Download Spacy model:**
```bash
python -m spacy download en_core_web_sm
```

**3. Configure your OpenAI API key**
[Best Practices for api key](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)

## Play game with LLM
> Now only arithmetic game under the zero-shot setting is implemented.
- This is the basic run command to evaluate the LLM in zero-shot setting. `GameName` can be choosed from [`arithmetic`,`sorting`,`mapreader`,`twc`]. `SetName` can be choosed from [`train`, `eval`, `dev`]. useSymbolicModules can be choosed from [`calc`,`navigation`,`sortquantity`,`kb-twc`]
  - To play arithmetic game, set `EnabledModules` to be `calc`.

```shell
python play_game_by_LLM.py --agent LLM --game_name ${GameName} --num_variations 100 --max_steps 20 --train_or_eval eval --set ${SetName} --useSymbolicModules ${EnabledModules}
```

```shell
# quickly start: evaluate LLM on dev set of arithmetic game and enable the calc symbolic module
python play_game_by_LLM.py --agent LLM --game_name arithmetic --num_variations 100 --max_steps 20 --train_or_eval eval --set dev --useSymbolicModules calc
```
- if you want to save data for training (the source-target style file), please add `--save-data`, and choose the desired game set.
- if you want to use `WandB` to log the histories, please add `--with-wandb`. For WandB configuration, please refer to `common.py`.
- for debug mode, add `--debug`, this will only play 2 episode of games, and turn off the wandb.


## Citing

If this AAAI 2024 paper is helpful in your work, please cite the following:

```
@inproceedings{zhong2023rspt,
  title={Large language models are neurosymbolic reasoners},
  author={Fang, Meng and Deng, Shilong and Zhang, Yudi and Shi, Zijing and Chen, Ling and Pechenizkiy, Mykola and Wang, Jun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```