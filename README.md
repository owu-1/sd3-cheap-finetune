# Work in progress - currently broken - cheap sd3 finetuning

## Use at your own risk. If unfamiliar with cloud computing, beware of unexpected charges

Warning: The training script uses a dataset which contains NSFW content.

There are three different types of instances in the NC A100 v4 series.

https://learn.microsoft.com/en-us/azure/virtual-machines/nc-a100-v4-series

| Size | vCPU | Memory (GiB) | Temp Disk (GiB)  | NVMe Disks | GPU | GPU Memory (GiB) |
|---|---|---|---|---|---|---|
| Standard_NC24ads_A100_v4   | 24  | 220 |64 | 960 GB | 1 | 80 |
| Standard_NC48ads_A100_v4   | 48 | 440 | 128| 2x960 GB| 2 | 160 |
| Standard_NC96ads_A100_v4   | 96 | 880 | 256| 4x960 GB | 4 | 320 |

The instance family is avaliable cheapest currently in westus3.

You must get the required quotas in these categories to use spot instances of this instance family:
- Total Regional vCPUs
- Total Regional Spot vCPUs
- Standard NCADS_A100_v4 Family vCPUs

For example, to get access to a Standard_NC24ads_A100_v4 spot instance in the westus3 region, you must have the following quotas limits:
- Total Regional vCPUs >= 24
- Total Regional Spot vCPUs >= 24
- Standard NCADS_A100_v4 Family vCPUs >= 24

A Standard_NC24ads_A100_v4 spot instance can be launched with the opentofu configuration in the folder opentofu. Some instructions to start the spot instance:
- Install the azure cli
- Login to the azure cli
- Create a RSA keypair (note Azure does not support ed25519)
- Edit env.sh to include the RSA public key path
- Enter the tofu environment ```bash env.sh```
- Init tofu ```tofu init```
- Apply tofu configuration ```tofu apply```

The following steps setup the instance for finetuning:
- SSH into the instance
- Setup the NVMe disk ```sudo bash nvme.sh```
- Setup pytorch environment ```bash setup_image_gen.sh```
- Apply datasets.patch to ```!conda-site-packages-datasets-dir!/packaged_modules/webdataset/webdataset.py``` e.g. ```patch /mnt/resource_nvme/miniconda3/envs/build/lib/python3.11/site-packages/datasets/packaged_modules/webdataset/webdataset.py < datasets.patch```

Run custom_train.py to start finetuning ```accelerate launch custom_train.py```

To take down the instance, comment out all lines in a100.tf and run ```tofu apply```
