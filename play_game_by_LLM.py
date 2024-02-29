import json
import os
import random
import time

import openai
from scienceworld import BufferedHistorySaver
from tqdm import tqdm
import numpy as np
import wandb
from common import parse_args
from main import (addModuleResultToInfo, initializeEnv, resetWithVariationDev,
                  resetWithVariationTest, resetWithVariationTrain,
                  sanitizeInfo)
from utils.symbolicModule import *


openai_api_key = os.environ.get("OPENAI_API_KEY")

def llm(task_description, prompt, stop=["\n"]):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are required to complete the task step by step. The task description is: {task_description}. Please always reply me with a valid action from the action set.\n\n".format(
              task_description=task_description)},
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message



def llm_multiple_round(prompt, stop=["\n"]): 
    i = 2
    while True:
        try:
            time.sleep(1)
            # uncomment for response from LLM
            # reason = "Please also give me a reason for your choice"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                temperature=0.5,
                top_p=0.5,
            )
            print(completion.choices[0].message)
            return completion.choices[0].message
        except openai.error.InvalidRequestError as e:
            print(e)
            length = len(prompt)
            cut_length_front = max(1, int(length / 2) -4)
            cut_length_back = int(length / 2) + 4
            prompt = prompt[:cut_length_front] + prompt[cut_length_back:]
        except:

            i = i * 2
            print("Error in LLM, sleep seconds: ", i)
            time.sleep(i)
            if i > 128:
                return {'content': ''}

# Collect data by running the LLM on the game
def CollectDatabyLLM(args):
    # TODO: add a few-shot learning option
    assert args.agent == "LLM"

    gameName = args.game_name
    enabledModules = args.useSymbolicModules
    setName = args.set

    agentName = args.agent

    args.real_fewshot = True
    real_fewshot_label = 'use_real_data' if args.real_fewshot else 'use_LLM_data'

    fewshot_label=f'fewshot_{real_fewshot_label}' if args.fewshot else 'zeroshot'

    agentName = f"{agentName}_{fewshot_label}" 

    # Configure output dir to save the results
    log_dir = os.path.join("logs", gameName, setName, enabledModules, agentName, args.timestamp if not args.debug else "debug")

    os.makedirs(log_dir, exist_ok=True)

    verboseFilenameOut = os.path.join(log_dir, "results.json")

    # Save Data
    if args.save_data:
        save_dir = os.path.join(args.data_save_path, 
                                gameName, 
                                setName,
                                enabledModules, 
                                agentName,
                                args.timestamp if not args.debug else "debug"
        )
        os.makedirs(save_dir, exist_ok=True)

        data_collection = []

    if args.fewshot:
        use_demo = True
        if args.real_fewshot:

            data_file_path = "data/LLM-gamearithmetic-numepisodes100.train.sourcetarget.json"
            with open(data_file_path) as file:
                data = [json.loads(line) for line in file]
            for index, single_step in enumerate(data):
                if 'complete' in single_step['target']:
                    continue
                separate_data = single_step['source'].split('</s>')
                obs = separate_data[1][5:]
                inv_state =  separate_data[2][19:-2]
                # act = data[index+1]['source'].split('</s>')[5][' PACT  ']
                act = single_step['target']
            demo_num = 3 * 6
            data = data[:demo_num]

            print('Source:', data['source'])
            print('Target:', data['target'])

            demo = """There are some examples:
            Trajectory 1:

            
            """
        raise NotImplementedError("Few-shot learning is not implemented yet.")


    if args.debug:
        args.with_wandb = False
    # WandB
    if args.with_wandb:
        from utils.wandb_utils import init_wandb
        args.wandb_group = f"{gameName}_{setName}_{enabledModules}_{agentName}"
        init_wandb(args)

    # Initialize the environment
    env = initializeEnv(threadNum=3, args=args)

    # Pick which set to evaluate on
    variations = []
    if (setName == "train"):
        variations = list(env.getValidSeedsTrain())
    elif (setName == "dev"):
        variations = list(env.getValidSeedsDev())
    elif (setName == "test"):
        variations = list(env.getValidSeedsTest())
    else:
        print("ERROR: Unknown set to evaluate on (" + str(setName) + ")")
        exit(1)

    # History saver
    bufferedHistorySaver = \
        BufferedHistorySaver(
            filenameOutPrefix=f"{log_dir}/histories",
        )

    # Log output prefix
    if (len(args.output_path) > 0):
        args.output_path = args.output_path + "/"

        # Make path if it doesn't exist
        if (not os.path.exists(args.output_path)):
            os.makedirs(args.output_path)

    scores = []
    totalSteps = []
    totalEnvSteps = []
    totalModuleSteps = []

    # Determine a (sub)set of variations to run
    maxVariations = args.num_variations
    if (len(variations) > maxVariations):
        print("NOTE: More than " + str(maxVariations) +
              " variations.  Only evaluating 100.")
        variations = variations[:maxVariations]

    unmatched_action_count = 0
    wandb_data = []

    if args.debug:
        variations = variations[:2]
    # TODO: make tqdm work for outputing the results
    for variationIdx in tqdm(variations):
        # Reset with this new variation(seed), based on the set
        obs = ""
        moduleInterface = None
        if (setName == "train"):
            info, moduleInterface = resetWithVariationTrain(
                env, args, variationIdx)
        elif (setName == "dev"):
            info, moduleInterface = resetWithVariationDev(
                env, args, variationIdx)
        elif (setName == "test"):
            info, moduleInterface = resetWithVariationTest(
                env, args, variationIdx)
        else:
            print("ERROR: Unrecognized set to evaluate on (" +
                  str(setName) + ")")
            exit(1)

        # Give modules initial observations
        print(type(info))
        print(info)

        moduleInterface.giveEnvironmentStatus(
            info['observation'], info['inventory'], info['look'])
        # Sanitize info, and add in module commands to valid actions
        lastRawInfo = info          # lastRawInfo should be unsanitized version?
        info = sanitizeInfo(info, moduleInterface)

        task_description = info['taskDescription']
        prev_obs = ''
        prev_action = ''

        done = False
        score = 0.0
        step = 0

        # The env has an internal step count, some actions like look around are free
        max_steps = args.max_steps 

        lastNActions = []
        history = []

        # Save initial observation
        info['stepsSinceLastReset'] = step
        history.append(info.copy())

        # Trying to set this up to match what the histories look like when they're saved, so the source->target between train/eval look identical.
        obs = info['obs']
        done = info['done']
        score = info['score']
        # This looks like a bug in the training code (that sets prev_obs to the same value as obs for the first iteration) -- but repeating it here.
        prev_obs = obs
        prev_action = ""
        actionToTake = ""

        # ! Modify the prompt template here
        prompt_template = "{observation}"
        prompt_template += "{inventory_state}\n"
        if prompt_template == "sorting":
            prompt_template += "\n\nYour current score is: {score}.\n"
        if gameName == "mapreader":
            prompt_template += "{subgoal}\n\n"
        else:
            prompt_template += ""
        prompt_template += "The valid action set contains: \n{valid_actions}.\n\n"


        if gameName == "sorting":
            prompt_template = "{task_description}\n"
            prompt_template += "Don't choose 'look around' for more than once.\n"
            prompt_template += "To sort the items one by one, please follow the instruction: \n"
            prompt_template += "1) choose 'sort ascending' or 'sort descending' to know the order to sort,\n"
            prompt_template += "2) take the items,\n"
            prompt_template += "3) put the items in box.\n"
        if gameName == "arithmetic":
            prompt_template += "DO NOT choose 'put math problem in box' "
        prompt_template += "Please choose one action from the valid action set to finish the task step by step. \n"
        prompt_template += "Do NOT respond with any other text, and you cannot decline to take an action.\n"
        # prompt_template += "Please also give a reason for your action. \n"
        prompt_history = ''

        prompt_history_template = "The following sentences describe your observation at timestep {timestep}: {observation}\n"
        prompt_history_template += "Your action at timestep {timestep}: {actionToTake}" + '\n'

        use_multi_round_chat = True
        if use_multi_round_chat:
            
            task_description = info['taskDescription']
            role_initialize = ''
            if gameName == "arithmetic":
                role_initialize = "You are a robot. {task_description}\n\n"
                role_initialize += "You are required to choose action from the valid action set to complete the task step by step."
                
            elif gameName == "sorting": 
                role_initialize = "You are a robot. {task_description} You need to follow the instructions to choose 'sort ascending' or 'sort descending' to call the external tool to help you play the game step by step.\n\n"
                role_initialize = "{task_description}\n\n"
                role_initialize += "You are required to choose action from the valid action set to complete the task step by step."
                role_initialize += "At the beginning of the game, you MUST choose 'look around' to get all the items.\n"
                role_initialize += "Before you take anything, you must choose 'sorting ascending' or 'sorting descending' to get the order of items to sort.\n"
                role_initialize += "To sort the items one by one, please follow the instruction: \n"
                role_initialize += "1) choose 'sort ascending' or 'sort descending' to know which one should be sort next. \n"
                role_initialize += "2) take the items,\n"
                role_initialize += "3) put the items in box.\n"
                role_initialize += "\n"
                role_initialize += "When you put the items into the box in the right order, you can get positive score. Please try to get as much score as possible.\n\n"

            elif gameName == "mapreader":
                coinLocation = re.search('located in the (.*)', task_description).group(1).split(",")[0].strip()
                boxLocation = re.search('the box found in the (.*)', task_description).group(1).split(".")[0].strip()
                role_initialize = "You are a navigation robot.\n" 
                role_initialize += "{task_description}\n"
                role_initialize += "Please finish your task as soon as possible.\n\n"
                role_initialize += "Every timestep, you are required to choose action from the valid action set to complete the task step by step."
                role_initialize += "You are encouraged to choose 'read map' to get the unknown surrounding layout.\n"
                role_initialize += "At the beginning choose 'read map' to get the unknown surrounding layout.\n"
                role_initialize += "After that, if you do not know how to get to SOMEPLACE, you can choose 'next step to SOMEPLACE' to get the path to SOMEPLACE.\n"
                role_initialize += "To choose the action, 'task', you can recall your task.\n"
                role_initialize += "Do NOT go to anywhere that is unnecessary for finishing the task.\n"
                role_initialize += "Please also give me a reason of your action.\n\n"

            elif gameName == 'twc':
                role_initialize += "You are a robot that are required to rearrange the items in the right place.\n"
                role_initialize += "To learn the right place of an item 'xxx', please choose 'query xxx'.\n"
                role_initialize += "When you take the item, you will get positive score.\n"
                role_initialize += "When you put the item in the right place, you will get higher positive score. Otherwise you get 0.\n"
                role_initialize += "You are supposed to get as much score as possible.\n\n"
            role_initialize += "To take action, respond with an action in the valid action set.\n"
            if gameName == "sorting" or gameName == "arithmetic":
                role_initialize += "There are some rules for choosing action: \n"
                role_initialize += "The next action of 'take math problem' is 'read math problem'. \n"
                role_initialize += "If you do not see the items that meet your requirements, please choose 'look around'.\n"
                role_initialize += "If you want to put something in the box, please first take it and then put it in box.\n"
                role_initialize += "For example, if you want to put 20 apples in the box, you should first choose 'take 20 apples' and then choose 'put 20 apples in box'.\n"
                role_initialize += "However, please never choose 'put math problem in box' as action. \n"
            role_initialize = role_initialize.format(task_description=task_description)

            chat_history = [{'role': 'system',
                             'content': role_initialize}]

        timestep = 0
        moduleResult_current_ep = None
        while not done:

            print("\n----------------------------------------------------------------------------------------------------\n")
            validActions = info['valid']

            prev_action = lastNActions[-1] if len(lastNActions) > 0 else ""

            task_description = info['taskDescription']
            current_observation = info['obs']
            inventory_state = info['inv'][14:]

            # Generate the next action
            if 'empty' not in inventory_state:
                inventory_state = "\nYour inventory contains: {inventory_state}.".format(inventory_state = inventory_state.replace('\n ', ',').replace('\n', ''))

            augmented_observation = current_observation

            if gameName in ['']:
                if moduleResult_current_ep is not None:
                    augmented_observation += \
                                f'\nThe must be helpful for your task: {moduleResult_current_ep}\n'
            elif gameName == 'twc':
                if moduleResult_current_ep is not None:
                    augmented_observation += \
                                f'\nThe must be helpful for your task: '
                    if not isinstance(moduleResult_current_ep, list):
                        moduleResult_current_ep = [moduleResult_current_ep]
                    augmented_observation += '\n '.join(moduleResult_current_ep)
                augmented_observation += '\n'
            subgoal = ''
            if gameName == 'mapreader':
                if 'coin' not in info['inv']:
                    subgoal = f"\nTo finish your task, now you need to go to {coinLocation} to take coin."
                    
                else:
                    subgoal = f"\nTo finish your task, now you need to go to {boxLocation} to put coin in box."
                    
            prompt = prompt_template.format(
                task_description=task_description,
                observation=augmented_observation,
                inventory_state=inventory_state,
                valid_actions="\n".join(validActions),
                subgoal=subgoal,
                score=info['score'],
                past_experinces='',
            )

            print("="* 20)
            print("* Prompt: " + prompt)
            print("="* 20)

            valid_flag = False
            count_try = 0
            while not valid_flag and count_try < 3:
                count_try += 1
                if use_multi_round_chat:
                    if count_try == 1:
                        chat_history.append(
                            {'role': 'user', 'content': prompt})
                    else:
                       
                        system_content = "The choosed action, '{actionToTake}', is not in the valid action set.\n"
                        system_content += "Please choose another action from the valid action set.\n"
                        system_content += "The valid action set is: {valid_actions}.\n"
                        system_content += "The current observation is {observation}.\n"
                        system_content += "Do NOT apology!\n"
                        if gameName == "mapreader":
                            if actionToTake.startswith("next step to"):
                                system_content = "If you can not choose '{actionToTake}', please read map first.\n"
                        system_content += "If you want to put something, please take it before.\n"
                        
                        system_content = system_content.format(
                            actionToTake=actionToTake,
                              observation=current_observation \
                                            if moduleResult_current_ep is None else current_observation + \
                                            f'The must be helpful for your task: {moduleResult_current_ep}',
                              valid_actions=", ".join(validActions))
                        chat_history.append(
                            {'role': 'user', 'content': system_content})
                    actionToTake = llm_multiple_round(chat_history)['content']
                    if args.with_wandb:
                        wandb_data.append([
                            variationIdx,  # env id
                            step,  # step
                            chat_history,  # prompt
                            actionToTake,  # completion
                        ])
                else:
                    actionToTake = llm(task_description, prompt)['content']
                # ! Modify the output to make it compatiable here
                actionToTake = actionToTake.split('\n')[0].lower() # TODO: maybe can be removed

                def check_valid(actionToTake, validActions):
                    print("check if valid:", actionToTake)
                    if actionToTake not in validActions:
                        for action_candidate in validActions:
                            action_candidate_lower = action_candidate.lower()
                            actionToTake_lower = actionToTake.lower()
                            if action_candidate_lower in actionToTake_lower:
                                actionToTake = action_candidate
                                return True, actionToTake
                        if 'take the math problem' in actionToTake:
                            actionToTake = 'take math problem'
                            print('modify [take the math problem] into [take math problem]')
                            return True, actionToTake
                        return False, actionToTake
                    else:
                        return True, actionToTake
                valid_flag, actionToTake = check_valid(
                    actionToTake, validActions)

                if valid_flag and gameName == 'arithmetic':
                    if 'div' in actionToTake:
                        # or 'sub' in actionToTake:
                        actionToTake_separated = actionToTake.split(' ')
                        actionToTake = actionToTake_separated[0] + ' ' + \
                            actionToTake_separated[2] + \
                            ' ' + actionToTake_separated[1]

                if count_try >= 3:
                    unmatched_action_count += 1

                    actionToTake = random.choice(validActions)
                    print('Warning: LLM can not output a valid action, so we use random action. We choose {}'.format(
                        actionToTake))
                if args.with_wandb:
                    # the last dimension is score and done
                    wandb_data[-1] += [count_try, valid_flag,
                                       actionToTake, -100, False]

            print("Best action: " + str(actionToTake))

            if not use_multi_round_chat:
                prompt_history += "\n" + prompt_history_template.format(
                    observation=current_observation,
                    actionToTake=actionToTake
                )
            else:
                chat_history.append(
                    {'role': 'assistant', 'content': actionToTake}
                )
            # Take a step in the environment
            # First, check to see if the command is intended for a module
            moduleWasRun, moduleResult = moduleInterface.runCommand(
                actionToTake)
            
            if (moduleWasRun == True):
                # Symbolic module was run -- add result to current 'info'
                info = addModuleResultToInfo(
                    lastRawInfo, moduleResult, actionToTake)
                moduleResult_current_ep = moduleResult
                lastRawInfo['lastActionStr'] = ""

            else:
                # Command was not intended for symbolic module -- run environment
                # New API -- now returns a tuple
                _, _, _, info = env.step(actionToTake)
                lastRawInfo = info

            # Give modules observations from environment
            moduleInterface.giveEnvironmentStatus(
                lastRawInfo['observation'], lastRawInfo['inventory'], lastRawInfo['look'])
            # Sanitize info, and add in module commands to valid actions
            info = sanitizeInfo(info, moduleInterface)

            # Store last observation/action
            prev_obs = obs

            obs = info['obs']
            reward = info['reward']
            done = info['done']
            score = info['score']

            if args.with_wandb:
                wandb_data[-1][-1] = done
                wandb_data[-1][-2] = score
                prediction_table = wandb.Table(columns=[
                                               'env_id', 'step', 'prompt', 'completion', 'count_try', 'if valid action', 'action', 'score', 'done'], data=wandb_data)
                wandb.log({'log': prediction_table})

            # Save history
            info['stepsSinceLastReset'] = step
            history.append(info.copy())

            print("Obs: " + obs)
            # print("Input string: " + str(input_str))
            print(
                f"Variation: {variationIdx}, Step: {step}, Score: {score}, Action: {actionToTake}")
            print("")
            step += 1
            if (step >= max_steps):
                print("Maximum steps exceeded (" + str(step) + ").")
                break
            if done:
                print("Received 'done' signal from environment.")
                break

            lastNActions.append(actionToTake)
            if (len(lastNActions) > 3):
                lastNActions = lastNActions[-4:]

            print("LastNActions: " + str(lastNActions))

            # Early stopping if we're in a loop
            if (len(lastNActions) >= 4):
                if (len(set(lastNActions)) == 1):
                    print(
                        "All actions in history are the same -- model is likely in a loop, stopping early.")
                    break

            timestep += 1

        # Store history
        # Get history internally (keeps track of module commands)
        finalScore = 0
        if (len(history) > 0):
            finalScore = history[-1]['score']

        runHistory = {
            'finalScore': finalScore,
            'history': history,
        }

        bufferedHistorySaver.storeRunHistory(
            runHistory, variationIdx, notes={'step': step})
        bufferedHistorySaver.saveRunHistoriesBufferIfFull(
            maxPerFile=args.maxHistoriesPerFile)

        # Save Data
        if args.save_data:
            data_collection.append(runHistory)

        # Save scores (clip negative scores to 0, for averaging)
        if (score < 0):
            score = 0.0
        scores.append(score)

        # Save total number of steps
        totalSteps.append(len(history))

        # Total number of environment steps
        envSteps = len([x for x in history if len(x['lastActionStr']) > 0])
        moduleSteps = len([x for x in history if len(x['moduleCommand']) > 0])
        totalEnvSteps.append(envSteps)
        totalModuleSteps.append(moduleSteps)

        print("Run completed...")
        print("Scores: " + str(scores), np.array(scores).mean())
        print("Steps: " + str(totalSteps))
        print("Steps (Env): " + str(totalEnvSteps))
        print("Steps (Mod): " + str(totalModuleSteps))
        time.sleep(2)

    # Episodes are finished -- manually save any last histories still in the buffer
    bufferedHistorySaver.saveRunHistoriesBufferIfFull(
        maxPerFile=args.maxHistoriesPerFile, forceSave=True)

    avgScore = sum(scores) / len(scores)
    print("Average score: " + str(avgScore))

    avgSteps = sum(totalSteps) / len(totalSteps)
    print("Average steps: " + str(avgSteps))

    avgEnvSteps = sum(totalEnvSteps) / len(totalEnvSteps)
    print("Average steps (env): " + str(avgEnvSteps))

    avgModSteps = sum(totalModuleSteps) / len(totalModuleSteps)
    print("Average steps (mod): " + str(avgModSteps))

    print('Bad Action Count:', unmatched_action_count)

    # Save to file
    scoresPacked = {
        'setName': setName,
        'lm': agentName,
        'gameName:': gameName,
        'enabledModules': enabledModules,
        'scores': scores,
        'totalSteps': totalSteps,
        'totalEnvSteps': totalEnvSteps,
        'totalModuleSteps': totalModuleSteps,
        'avgScore': avgScore,
        'avgSteps': avgSteps,
        'avgEnvSteps': avgEnvSteps,
        'avgModSteps': avgModSteps,
        'numSamples': len(scores),
    }

    if args.save_data:
        from main import mkSourceTargetStrsFromHistory
        print(json.dumps(data_collection, indent=4, sort_keys=True))

        # Step 1: Convert the histories to source/target strings
        sourceTargetOut = []
        for oneRun in data_collection:
            print("----")
            sourceTargetStrs = mkSourceTargetStrsFromHistory(oneRun, args)
            print(json.dumps(sourceTargetStrs, indent=4, sort_keys=True))
            sourceTargetOut.extend(sourceTargetStrs)

        # Step 2: Export source/target out to file
        numEpsiodes = len(data_collection)
        filenameOut = os.path.join(save_dir, str(
            numEpsiodes) + "eps" + ".sourcetarget.json")
        print("Exporting JSON lines file for T5 trainer: " + filenameOut)
        fp = open(filenameOut, "w")
        for stOut in sourceTargetOut:
            fp.write(json.dumps(stOut) + "\n")
        fp.close()
        print("Export completed.")

    if args.with_wandb:
        wandb.log(
            {
                "avgScore": avgScore,
                "avgSteps": avgSteps,
                "avgEnvSteps": avgEnvSteps,
                "avgModSteps": avgModSteps,
                "BadActionCount (not sure)": unmatched_action_count,
            }
        )
    print("Saving " + str(verboseFilenameOut))
    with open(verboseFilenameOut, "w") as write_file:
        json.dump(scoresPacked, write_file, indent=4)

    print("Completed.")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    CollectDatabyLLM(args)
