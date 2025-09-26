from tensorboard.backend.event_processing import event_accumulator
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

runs = []
behaviors = [[112], [29, 1], [23, 8, 21, 25], [90, 60, 4, 7, 88, 123, 118, 38], [23, 46, 29, 3, 128, 124, 63, 16, 119, 17, 22, 131, 11, 69, 77, 123]]
training_type = '_el32a'

scalars = ["eval/loss", "eval/outputs/accuracies", "train/loss", "train/outputs/accuracies", "train/margins", "eval/margins"]

for behavior in behaviors:
    train_log_files = [join('logs/gen_behavior_' + str(behavior) + training_type, f) for f in listdir('logs/gen_behavior_' + str(behavior) + training_type) if isfile(join('logs/gen_behavior_' + str(behavior) + training_type, f))]
    ea = event_accumulator.EventAccumulator(train_log_files[-1])
    ea.Reload()
    print(ea.Tags())
    print(behavior)
    data = {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    runs.append(data)

plasma = cm = plt.get_cmap('plasma')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)

fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    epochs = [t for t in runs[i][scalars[0]]["step"]]
    axs.plot(epochs[:100], runs[i]["train/loss"]["value"][:100], color=colorVal, label=str(2**i))
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Training Loss', fontsize=18)
axs.set_title('Training Loss Curves', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig('last_layer_figures/' + training_type[1:] + '_train_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs[:100], runs[i]["eval/loss"]["value"][:100], color=colorVal, label=str(2**i))
    print(runs[i]["eval/loss"]["value"][99])
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Test Loss', fontsize=18)
axs.set_title('Test Loss Curves', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig('last_layer_figures/' + training_type[1:] + '_test_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs[:100], runs[i]["train/outputs/accuracies"]["value"][:100], color=colorVal, label=str(2**i))
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Training Accuracy', fontsize=18)
axs.set_title('Training Accuracy', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig('last_layer_figures/' + training_type[1:] + '_train_acc_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs[:100], runs[i]["eval/outputs/accuracies"]["value"][:100], color=colorVal, label=str(2**i))
    print(runs[i]["eval/outputs/accuracies"]["value"][99])
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Test Accuracy', fontsize=18)
axs.set_title('Test Accuracy', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig('last_layer_figures/' + training_type[1:] + '_test_acc_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs[:100], runs[i]["train/margins"]["value"][:100], color=colorVal, label=str(2**i))
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Training Reward Margin', fontsize=18)
axs.set_title('Training Reward Margin', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig('last_layer_figures/' + training_type[1:] + '_train_margin_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    axs.plot(epochs[:100], runs[i]["eval/margins"]["value"][:100], color=colorVal, label=str(2**i))
    print(runs[i]["eval/margins"]["value"][99])
axs.set_xlabel('Step', fontsize=18)
axs.set_ylabel('Test Reward Margin', fontsize=18)
axs.set_title('Test Reward Margin', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig('last_layer_figures/' + training_type[1:] + '_test_margin_verify.pdf')


Ks = [1, 2, 4, 8, 16]
margins = []
accs = []

for i in range(len(behaviors)):
    margins.append(runs[i]["eval/margins"]["value"][99])
    accs.append(1-runs[i]["eval/outputs/accuracies"]["value"][99])

fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))
axs.plot(Ks, margins, color=scalarMap.to_rgba(0.4))
axs.set_xlabel('Concept Set Size', fontsize=18)
axs.set_ylabel('Test Reward Margin', fontsize=18)
axs.set_title('Test Reward Margin vs. Concepts', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
#axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig('last_layer_figures/' + training_type[1:] + '_test_margin_clusters_verify.pdf')

print(accs)
fig, axs = plt.subplots(1, 1, figsize=(6.5, 4))
axs.plot(Ks, accs, color=scalarMap.to_rgba(0.4))
axs.set_xlabel('Concept Set Size', fontsize=18)
axs.set_ylabel('Test Error', fontsize=18)
axs.set_title('Test Error vs. Concepts', fontsize=24)
handles,labels = axs.get_legend_handles_labels()
#axs.legend(handles, labels, loc='lower right', title='Clusters', fontsize=14, title_fontsize=16)
plt.tight_layout()
plt.savefig('last_layer_figures/' + training_type[1:] + '_test_acc_clusters_verify.pdf')