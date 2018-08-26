# Code for the Transformer and the semi-autoregressive Transformer (SAT)
The is the official implementation of the paper: **Semi-Autoregressive Neural Machine Translation**, which will be presented at EMNLP 2018.

## Contact
Email: <chqiwang@126.com>

## Prepare Data
TBD

## Usage

### Step 1: Train the base Transformer model
Train the Transformer

	python train.py -c configs/transformer.yaml

Then average the last five checkpoints

	python third_party/tensor2tensor/avg_checkpoints.py --prefix "model-transformer/model_step_" --checkpoints 96000,97000,98000,99000,100000 --output_path "model-transformer/model_avg"


### Step 2: Translate source sentences in the corpus into the target language
	python evaluate.py -c configs/transformer.yaml

### Step 3: Train and evaluate the SAT
(Plsease replace [K] in the following commands with an integer larger than one)

Copy the base model

	cp -r model-transformer model-sat-[K]

Train the SAT

	python train.py -c configs/sat-[K].yaml

Then average the last five checkpoints

	python third_party/tensor2tensor/avg_checkpoints.py --prefix "model-sat-[K]/model_step_" --checkpoints 96000,97000,98000,99000,100000 --output_path "model-sat-[K]/model_avg"

Evaluate the model

	python evaluate.py -c configs/sat-[K].yaml

## Notes
* Each steps will takes a long time.
* By default, we use 8 gpu devices when train and predict. If you have less then 8 gpus, you should modify the yaml config files.
