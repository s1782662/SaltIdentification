from model import *
from SegmentationDataGenerator import SegmentationDataGenerator
from albumentations import (HorizontalFlip,ShiftScaleRotate,RandomContrast,RandomBrightness,Compose)
from augmentations import get_augmentations
import pandas as pd
import numpy as np
from params import args

# Read the folds
train = pd.read_csv(args.folds_csv)
df_train = train[train.fold != fold].copy().reset_index(drop=True)
df_valid = train[train.fold == fold].copy().reset_index(drop=True)
ids_train,ids_valid = df_train[df_train.unique_pixels > 1].id.values,df_valid[df_valid.unique_pixels > 1].id.values

augs = get_augmentations('valid',1.0)
dg = SegmentationDataGenerator(input_shape=(args.input_size,args.input_size),
                            batch_size=args.batch_size,
                            augs=augs,
                            preprocess=None
                            )

train_generator = dg.train_batch_generator(ids_train)
valid_generator = dg.evaluation_batch_generator(ids_valid)

model = Unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5',monitor='loss',verbose=1,save_best_only=True)


model.fit_generator(generator=train_generator,
                    steps_per_epoch = ids_train.shape[0] // args.batch_size * 2,
                    epochs=args.epochs,
                    callbacks=[model_checkpoint],
                    validation_data = valid_generator,
                    validation_steps=np.ceil(ids_valid.shape[0]/ args.batch_size))
