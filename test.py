import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from BrainTumorDataset import BrainSegmentationDataset as Dataset
from model import UNet as Net
from utils import dsc, gray2rgb, outline


predictions = 'predictions'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
images = 'kaggle_3m'
image_size = 256
batch_size = 32
weights = 'weights\model.pt'

os.makedirs(predictions, exist_ok=True)
dataset = Dataset(images_dir=images, subset="validation", image_size=image_size, random_sampling=False)
loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=1)

with torch.set_grad_enabled(False):
    model = Net(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    input_list = []
    pred_list = []
    true_list = []
    
    for i, data in tqdm(enumerate(loader)):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)

        y_pred = model(x)
        y_pred_np = y_pred.detach().cpu().numpy()
        pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

        y_true_np = y_true.detach().cpu().numpy()
        true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

        x_np = x.detach().cpu().numpy()
        input_list.extend([x_np[s] for s in range(x_np.shape[0])])
        
        
        
volumes = {}
num_slices = np.bincount([p[0] for p in loader.dataset.patient_slice_index])
index = 0
for p in range(len(num_slices)):
    volume_in = np.array(input_list[index : index + num_slices[p]])
    volume_pred = np.round(np.array(pred_list[index : index + num_slices[p]])).astype(int)
    volume_pred = largest_connected_component(volume_pred)
    volume_true = np.array(true_list[index : index + num_slices[p]])
    volumes[loader.dataset.patients[p]] = (volume_in, volume_pred, volume_true)
    index += num_slices[p]
    
    
dsc_dict = {}
for p in volumes:
    y_pred = volumes[p][1]
    y_true = volumes[p][2]
    dsc_dict[p] = dsc(y_pred, y_true, lcc=False)
    

y_positions = np.arange(len(dsc_dict))
dsc_dict = sorted(dsc_dict.items(), key=lambda x: x[1])
values = [x[1] for x in dsc_dict]
labels = [x[0] for x in dsc_dict]
labels = ["_".join(l.split("_")[1:-1]) for l in labels]
fig = plt.figure(figsize=(12, 8))
canvas = FigureCanvasAgg(fig)
plt.barh(y_positions, values, align="center", color="skyblue")
plt.yticks(y_positions, labels)
plt.xticks(np.arange(0.0, 1.0, 0.1))
plt.xlim([0.0, 1.0])
plt.gca().axvline(np.mean(values), color="tomato", linewidth=2)
plt.gca().axvline(np.median(values), color="forestgreen", linewidth=2)
plt.xlabel("Dice coefficient", fontsize="x-large")
plt.gca().xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
plt.tight_layout()
canvas.draw()
plt.close()
s, (width, height) = canvas.print_to_buffer()
dsc_dist_plot =  np.fromstring(s, np.uint8).reshape((height, width, 4))

#imsave('dsc.png', dsc_dist_plot)

for p in volumes:
    x = volumes[p][0]
    y_pred = volumes[p][1]
    y_true = volumes[p][2]
    for s in range(x.shape[0]):
        image = gray2rgb(x[s, 1])  # channel 1 is for FLAIR
        image = outline(image, y_pred[s, 0], color=[255, 0, 0])
        image = outline(image, y_true[s, 0], color=[0, 255, 0])
        filename = "{}-{}.png".format(p, str(s).zfill(2))
        filepath = os.path.join(predictions, filename)
        imsave(filepath, image)
    
    

