# Learn to play symbolic games with LLM guidances
This repository is implemented based the repo for the paper ["Behavior Cloned Transformers are Neurosymbolic Reasoners"](https://arxiv.org/abs/2210.07382) (EACL 2023).

## Before using the repo

**1. Install Dependencies:**
```bash
conda create --name t5-neurosymbolic python=3.9
conda activate t5-neurosymbolic
pip install -r requirements.txt
```
You may want to install the pytorch manually if your GPU does not support CUDA 11.

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
  - [ ] **Not Implement yet.** To play other games, xxx

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
- [ ] **Not Implement yet.** if you want to use fewshot setting, please add `--fewshot`.


## Replicating the experiments
**1. Clone the repository:**
```bash
git clone git clone https://github.com/cognitiveailab/neurosymbolic.git
cd neurosymbolic
```

**2. Install Dependencies:**
```bash
conda create --name t5-neurosymbolic python=3.9
conda activate t5-neurosymbolic
pip install -r requirements.txt
```
You may want to install the pytorch manually if your GPU does not support CUDA 11.

**3. Download Spacy model:**
```bash
python -m spacy download en_core_web_sm
```


**4. Train the T5 model:**
Use the Huggingface Seq2seq trainer to train a T5 model using the pre-generated training data.  We've provided runscripts in the `runscripts` folder for training each of the 4 games, in each of the two modes (with/without modules).  The runscript takes a single paramter, the number of training epochs.  For example, to train the T5 model for the Arithmetic game, with symbolic modules, the command would be:
```bash
runscripts/runme-train-arithmetic.sh 4
```

This will save the trained model in a verbosely named folder (e.g. `t5twx-game-arithmetic-withcalcmodule-base-1024-ep4`).


**5. Evaluate the performance of that model on unseen variations of the game:**

We'll use the above example as a follow-through:

`python main.py --game_name=arithmetic --num_variations 100 --max_steps=20 --train_or_eval=eval --set=dev --lm_path=t5twx-game-arithmetic-withcalcmodule-base-1024-ep4  --useSymbolicModules=calc`

This will generate two verbosely-named output logs:
- `resultsout-arithmetic-modcalccalc-lmt5twx-game-arithmetic-withcalcmodule-base-1024-ep4-setdev.json`: a summary file describing the overall scores for each game variation.
- `t5saveout-gamearithmetic-lmt5twx-game-arithmetic-withcalcmodule-base-1024-ep4-dev-10000-10099.json`: a detailed file with the complete play logs of each variation.


**Running different games/modes:**
The relevant parameters when calling main are:
- **game_name**: one of: `arithmetic, mapreader, sorting, twc-easy`
- **lm_path**: the name of the trained model from Step 4
- **useSymbolicModules**: one of: `calc, navigation, sortquantity, kb-twc`

See the file [runscripts/runme-tuning-batch-with-module.sh](runscripts/runme-tuning-batch-with-module.sh) for more examples.


# Frequently Asked Questions (FAQ)
**Q: What symbolic modules were implemented in this work?**

A: (1) A *calculator* for arithmetic (+, -, *, /). (2) A *navigation* module that helps perform pathfinding by providing the next location to traverse to move closer towards a destination. (3) A *knowledge base lookup*, that provides a list of triples that match a query term.  (4) A *sorting* module, that sorts lists of quantities in ascending order, and is unit-aware (e.g. 50 milligrams is less than 10 grams).

**Q: What model size did you use?**

A: As we note in the broader impacts, one of the benefits of augmenting language models with symbolic modules is that you can use smaller models and still get good performance.  For the experiments reported here we used T5-base (220M parameter) model.

**Q: How long does it take to train and evaluate the models?**

A: As an example, it generally takes under 10 minutes on an RTX 4090 to train the T5-base model that uses symbolic modules for the Arithmetic game when training up to 4 epochs.  Evaluting each of the 100 variations in the dev or test set takes a little longer, but still on the order of about 10 minutes.

**Q: What was the tuning procedure?**

A: Behavior Cloned Transformer model performance was tuned on a single hyperparameter (number of training epochs), tuned from 2 to 20 epochs in 2 epoch increments.  There are islands of high and low performance, so it's highly recommended to tune.

**Q: How much training data is required?**

A: While we didn't run an analysis of performance versus training data size for the EACL paper, it does not appear as though much data is required for this technique to work.  For all games, models are trained on 100 training games, which (nominally) corresponds to 100 examples of the symbolic module being used.  For the arithmetic game where there are technically four different module actions being used (add, sub, mul, div), the model sees each action only 25 times during training, and still achieves near perfect performance.  All of these results are on T5-base, a 220M parameter model, and the model has no pretraining on the text game (i.e. it has to learn to play the game, and use the symbolic module at the same time).  This suggests modest amounts of training data are required for the model to learn to use these symbolic modles, at least for the games explored in this paper.

**Q: What information is packed into the prompt of the T5 agent?**

A: The T5 agent prompt is essentially a first order Markov model -- it sees information from the current step, and the previous step (longer is generally not possible due to the token length limitations of the input).  The specific packing here is:
`promptStr = task_description + ' </s> OBS ' + cur_obs + ' </s> INV ' + cur_inv + ' </s> LOOK ' + cur_look + ' </s> <extra_id_0>' + ' </s> PACT ' + prev_action + ' </s> POBS ' + prev_obs + ' </s>' `.

Where `task_description` is the game task description (so the model knows the overall task it's supposed to complete), `cur_obs` is the latest observation string from the text game or symbolic module, `cur_inv` is the current agent inventory, and `cur_look` is equivalent to the "look" action in the game at the current time step.  `prev_action` is the last action the agent chose, and `prev_obs` is the previous observation, which will be from the symbolic module if the previous action was for a symbolic module, and otherwise from the text game engine.

**Q: What text games did you use?**

A: We use 3 games made for this paper (arithmetic, mapreader, and sorting), and one existing benchmark text game called [TextWorld Common Sense (Murugesan et al., AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17090).  All games were implemented using [TextWorldExpress (repo)](https://github.com/cognitiveailab/TextWorldExpress), a new and extremely fast engine for simulating text games in natural language processing research ([also accepted to EACL as a demo paper (PDF)](https://arxiv.org/abs/2208.01174)).  TextWorldExpress is `pip` installable, and you can generally be up and running in minutes.

**Q: Why use text games as an evaluation?**

A: Text games are an exciting new research paradigm for evaluating multi-step reasoning in natural language processing.  They require models to perform explicit multi-step reasoning by iteratively observing an environment and selecting the next action to take.  This generally requires combining a variety of common-sense/world knowledge as well as task-specific knowledge to solve.  As we show in another paper ([ScienceWorld (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.775/)), models that perform extremely well on question answering tasks perform very poorly when those same tasks are reframed as text games, that explicitly test multi-step/procedural knowledge.  Here's a [survey paper on using text games in natural language processing](https://aclanthology.org/2022.wordplay-1.1/), that provides a quick overview of the field.

**Q: I have a question, comment, or problem not addressed here.**

A: We're committed to supporting this work.  Please create a github issue (which should send us an e-mail alert), or feel free to e-mail `pajansen@arizona.edu` if you prefer.

## ChangeLog/Other Notes
- Feb-23-2023: Updated to latest TextWorldExpress API for release, so we can just use the pip installable version of TWX.  Please post any issues.
- Feb-23-2023: Cleaned up repository for release, moving many files.  Please report any broken links/issues.


## Citing

If this EACL 2023 paper is helpful in your work, please cite the following:

```
@article{wang2022neurosymbolicreasoners,
  title={Behavior Cloned Transformers are Neurosymbolic Reasoners},
  author={Wang, Ruoyao and Jansen, Peter and C{\^o}t{\'e}, Marc-Alexandre and Ammanabrolu, Prithviraj},
  journal={arXiv preprint arXiv:2210.07382},
  year={2022},
  url = {https://arxiv.org/abs/2210.07382},
}

```
(Will update with EACL 2023 bibtex when available)
