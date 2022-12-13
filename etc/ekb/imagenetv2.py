"""
There is an issue with the naming convention after downloading imagenet v2
highlighted here
#https://github.com/modestyachts/ImageNetV2/issues/6


"""

import os, glob

DIR ="/datahd3a/datasets/pytorch_datasets/imagenetv2/imagenetv2*"

def main():
    for path in glob.glob(DIR):
        if os.path.isdir(path):
            for subpath in glob.glob(f'{path}/*'):
                print(subpath)
                dirname = subpath.split('/')[-1]
                os.rename(subpath, '/'.join(subpath.split('/')[:-1]) + '/' + dirname.zfill(4))

if __name__ == '__main__':
    main()

#%%
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#%%