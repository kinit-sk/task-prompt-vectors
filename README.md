# Task prompt vectors
This repository serves as supplementary material to our experiments with Task Prompt Vectors.

### [Supplementary material](https://github.com/kinit-sk/task-prompt-vectors/tree/main/supplementary)

### [Results and pre-trained weights](https://drive.google.com/file/d/1m9L94sugJNiQ5-c_KVphiO6cdPuye1iu/view?usp=sharing)

## Install requirements
`pip install -r requirements.txt`

## Download pre-trained soft-prompts and taks prompt vectors
Download from [Results and pre-trained weights](https://drive.google.com/file/d/1m9L94sugJNiQ5-c_KVphiO6cdPuye1iu/view?usp=sharing) and extract in current directory.

## Run and eval cross-origin experiments
Theese are the results in Table 1.
`python run_cross_origin [config_path]`
`python eval_cross_origin [config_path]`

## Run addition experiments
Theese are the results in Figure 2.
`python eval_addition.py`

## Run fewshot Experiments
Theese are the results in Figure 3.
`./run_fewshot [nli, cls, sent]`

## Run Prompt Tuning
`python prompt_tuning.py {path_to_config}`

## Generate figures
`python scripts/visual [script_name].py [path_to_results]`

## Contact
In case of any questions, contact [robert.belanec@kinit.sk](mailto:robert.belanec@kinit.sk)
