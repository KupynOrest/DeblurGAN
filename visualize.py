import seaborn as sns
sns.set_style('white')

import json
import numpy as np

with open("pretrain losses - srgan.json") as f:
    data = json.load(f)

print("Data loaded.")

# plot the generator loss values
sns.plt.plot(data['generator_loss'])
sns.plt.show()

print("Mean gan loss :", np.mean(data['generator_loss']))
print("Std gan loss : ", np.std(data['generator_loss']))
print("Min gan loss : ", np.min(data['generator_loss']))

# plot the PSNR loss values
sns.plt.plot(data['val_psnr'])
sns.plt.show()

print("Mean psnr loss :", np.mean(data['val_psnr']))
print("Std psnr loss : ", np.std(data['val_psnr']))
print("Min psnr loss : ", np.min(data['val_psnr']))

with open("pretrain losses - discriminator.json") as f:
    data = json.load(f)

print("Data loaded.")

# plot the discriminator loss values
sns.plt.plot(data['discriminator_loss'])
sns.plt.show()

print("Mean discriminator loss :", np.mean(data['discriminator_loss']))
print("Std discriminator loss : ", np.std(data['discriminator_loss']))
print("Min discriminator loss : ", np.min(data['discriminator_loss']))