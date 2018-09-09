# Code for the Transformer and the semi-autoregressive Transformer (SAT)
The is the official implementation of the paper: **Semi-Autoregressive Neural Machine Translation**, which will be presented at EMNLP 2018.

## Contact
Email: <chqiwang@126.com>

## Prepare
Clone this project

	git clone https://github.com/chqiwang/sa-nmt.git
	cd sa-nmt

Download **data.zip** from [google drive](<https://drive.google.com/open?id=1RWvAZfttwUQFGY76ItQ9Mmplq-RTWRm7>) then decompress it. Make sure the **data** folder is a subfolder of **sa-nmt**.

## Usage

### Step 1: Train the base Transformer model
Train the Transformer

	python train.py -c configs/transformer.yaml

Then average the last five checkpoints

	python third_party/tensor2tensor/avg_checkpoints.py --prefix "model-transformer/model_step_" --checkpoints 96000,97000,98000,99000,100000 --output_path "model-transformer/model_avg"


### Step 2: Translate source sentences in the corpus into the target language
	python evaluate.py -c configs/transformer.yaml

### Step 3: Train and evaluate the SAT
(Plsease replace [K] in the following commands with 2, 4 or 6)

Copy the base model

	cp -r model-transformer model-sat-[K]

Train the SAT

	python train.py -c configs/sat-[K].yaml

Then average the last five checkpoints

	python third_party/tensor2tensor/avg_checkpoints.py --prefix "model-sat-[K]/model_step_" --checkpoints 96000,97000,98000,99000,100000 --output_path "model-sat-[K]/model_avg"

Evaluate the model

	python evaluate.py -c configs/sat-[K].yaml

## Notes
* Each steps will takes a long time (maybe one day, depend on your gpu device).
* By default, we use 8 gpu devices when train and predict. If you have less then 8 gpus, you should modify the yaml config files (num_gpus, batch_size and tokens_per_batch).
* Raise an issue or email me if you have problem.
