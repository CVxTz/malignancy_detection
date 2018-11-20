from glob import glob

import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.applications.nasnet import preprocess_input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from models import get_model_classif_nasnet
from utils import get_id_from_file_path, data_gen, chunker

df_train = pd.read_csv("../input/train_labels.csv")
id_label_map = {k: v for k, v in zip(df_train.id.values, df_train.label.values)}
df_train.head()
labeled_files = glob('../input/train/*.tif')
test_files = glob('../input/test/*.tif')

train, val = train_test_split(labeled_files, test_size=0.1, random_state=101010)

model = get_model_classif_nasnet()

batch_size = 32
h5_path = "model.h5"
checkpoint = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

_ = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=True),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=2, verbose=1,
    callbacks=[checkpoint],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size)
batch_size = 64
_ = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=True),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=6, verbose=1,
    callbacks=[checkpoint],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size)

model.load_weights(h5_path)

preds = []
ids = []

for batch in chunker(test_files, batch_size):
    X = [preprocess_input(cv2.imread(x)) for x in batch]
    ids_batch = [get_id_from_file_path(x) for x in batch]
    X = np.array(X)
    preds_batch = ((model.predict(X).ravel() * model.predict(X[:, ::-1, :, :]).ravel() * model.predict(
        X[:, ::-1, ::-1, :]).ravel() * model.predict(X[:, :, ::-1, :]).ravel()) ** 0.25).tolist()
    preds += preds_batch
    ids += ids_batch

df = pd.DataFrame({'id': ids, 'label': preds})
df.to_csv("baseline_nasnet.csv", index=False)
df.head()
