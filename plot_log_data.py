from tensorboard.backend.event_processing import event_accumulator
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

train_runs = []
test_runs = []
behaviors = [[112], [29, 1], [23, 8, 21, 25], [90, 60, 4, 7, 88, 123, 118, 38], [23, 46, 29, 3, 128, 124, 63, 16, 119, 17, 22, 131, 11, 69, 77, 123]]
training_type = '_5.52.0'
model_type = 'Llama-3.1-8B-'

scalars = ["eval/loss", "eval/rewards/accuracies", "eval/rewards/margins"]

for behavior in behaviors:
    train_log_files = [join(model_type + 'logs/behavior_' + str(behavior) + training_type + "_0t", f) for f in listdir(model_type + 'logs/behavior_' + str(behavior) + training_type + "_0t") if isfile(join(model_type + 'logs/behavior_' + str(behavior) + training_type + "_0t", f))]
    print(train_log_files)
    ea = event_accumulator.EventAccumulator(train_log_files[-1])
    ea.Reload()
    data = {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    train_runs.append(data)

for behavior in behaviors:
    test_log_files = [join(model_type + 'logs/behavior_' + str(behavior) + training_type + "_0e", f) for f in listdir(model_type + 'logs/behavior_' + str(behavior) + training_type + "_0e") if isfile(join(model_type + 'logs/behavior_' + str(behavior) + training_type + "_0e", f))]
    print(test_log_files)
    ea = event_accumulator.EventAccumulator(test_log_files[-1])
    ea.Reload()
    data = {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    test_runs.append(data)

plasma = cm = plt.get_cmap('plasma')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    epochs = [t*128./1000. for t in train_runs[i][scalars[0]]["step"]]
    axs.plot(epochs, train_runs[i]["eval/loss"]["value"], color=colorVal, label=str(2**i))
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Training Loss', fontsize=18)
axs.set_title('Training Loss Curves', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_train_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs, test_runs[i]["eval/loss"]["value"], color=colorVal, label=str(2**i))
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Test Loss', fontsize=18)
axs.set_title('Test Loss Curves', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_test_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs, train_runs[i]["eval/rewards/accuracies"]["value"], color=colorVal, label=str(2**i))
    print("TRAIN ACC")
    print(train_runs[i]["eval/rewards/accuracies"]["value"][59])
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Training Accuracy', fontsize=18)
axs.set_title('Training Accuracy', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_train_acc_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs, test_runs[i]["eval/rewards/accuracies"]["value"], color=colorVal, label=str(2**i))
    print("TEST ACC")
    print(test_runs[i]["eval/rewards/accuracies"]["value"][59])
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Test Accuracy', fontsize=18)
axs.set_title('Test Accuracy', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_test_acc_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs, train_runs[i]["eval/rewards/margins"]["value"], color=colorVal, label=str(2**i))
    print("TRAIN MARGIN")
    print(train_runs[i]["eval/rewards/margins"]["value"][59])
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Training Reward Margin', fontsize=18)
axs.set_title('Training Reward Margin', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_train_reward_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs, test_runs[i]["eval/rewards/margins"]["value"], color=colorVal, label=str(2**i))
    print("TEST MARGIN")
    print(test_runs[i]["eval/rewards/margins"]["value"][59])
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Test Reward Margin', fontsize=18)
axs.set_title('Test Reward Margin', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_test_reward_verify.pdf')