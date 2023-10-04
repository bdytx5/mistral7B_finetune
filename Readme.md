This repo stores the code for a tutorial for fine tuning Mistral 7B on your own personal ChatGPT conversations as well as on Python Code!!!!

For training Mistral on Python Code, see train/alpaca-python-10k



In data/ you will find a script that allows you to generate a train and validation datasets from your ChatGPT data export

run it using 'python gen_json_ds.py val_pct conversations.json' where val_pct is your validation percentage and conversations.json 
is the full path to your ChatGPT data 

After generating your dataset, move the tr/val jsonl files over to the train directory, and specify their relative paths in the train script 

In train/ you will find a script for training the model, along with a choline.yaml file 

Choline is for easy deployment of ML workloads into the cloud (right now Vast.ai)

Before using Choline, you will need to setup a vast.ai account and add set up rsa keys on your machine. 
For a guide on that, see https://vast.ai/docs/gpu-instances/ssh

The Choline config specifies things like your hardware/software env requirements and data that you would like to send to the instance

You will want to replace the upload_locations  with the full path to your train directory (mine is /Users/brettyoung/Desktop/mistral7b/train)

Also, add your wandb api key after the "export WANDB_API_KEY=" in the setup script within the choline.yaml

You can also add more disk space / change GPU's / increase CPU Ram in the choline.yaml 

For the sake of the tutial, I already have created a choline.yaml file for you to use, however, you could 
create your own using the init.py script in choline/simple_startup 

The Choline library allows for ...
-finding cheap instances quickly (note, always check prices for both GPU cost AND internet upload / download costs (shoot for <.01)
-launching instances with a similar evironment to your local machine directly from the command line 'python simple_startup.py'
-syncing data in the cloud with your local machine easy, just run 'python sync.py'
-monitoring run status, just run 'python status.py' 
-wandb integrations -- specify your api key in choline.yaml setup script and you are good to go 


The choline.yaml file is somewhat self descriptive, but questions / suggestions are welcome 


Im sure Im not taking advantage of all of the possible speedups like flash attention etc etc and those will be for the next tutorial 

The current training config uses gradient checkpointing, so I've reserved a good amount of CPU ram to avoid overflows issues 
--Note errors like "exit code -9" or "exit code -XYZ" often signify CPU RAM memory issues in Docker

Im always open to new pull requests for configs that allow for more efficient training!!! 




If you would like to run this repo without choline, it should be pretty simple 
simply clone this repo on your instance, install pip dependencies, and run 'deepspeed train.py' 




Some notes... 
- Saving your model checkpoints requires A LOT of storage, keep this in mind when specifying storage amounts 
- errors like "exit code -XYZ" often signify CPU RAM memory issues in Docker. they can be solved with more CPU RAM 
- be mindful not only of GPU costs but upload/download costs on Vast.ai, as these can add up quickly 
- the dataset format is sequences of len=MAX_TOKENS where the sequence ends with a reponse from the assistant and begins 
  with maximum previous context (eg previous conversation data)
- Mistral specifies their format similar to this 

""
text = "<s>[INST] What is your favourite condiment? [/INST]"
"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "

"" 

-there is a bit of a delay when logging the model. This could be a bug somewhere or an issue with my code. 




Future goals .... 
- support for testing the model in a UI similar to chatgpt 
- further optimizing training for more efficiency 
