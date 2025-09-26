# dpo-diverse

DPO_training.py is the script for full fine-tuning. Details on arguments are provided in the file. 
An example script of how to use DPO_training.py is provided in example_train.sh. 

last_layer_training.py is the script for last-layer training. The file's current code is setup for Llama-3.1-8B and can be modified to be used for Llama-2-7B.
The file can run directly with python.

persona_structure.ipynb generates all the data needed to verify the cluster assumption as done in the paper.

plot_log_data.py and plot_last_layer.py are the plotting scripts for the logs of full fine-tuning and last-layer training respectively.

persona_selection.py is the script for selecting the personas to use with the current seed corresponding to the setup in the paper.
