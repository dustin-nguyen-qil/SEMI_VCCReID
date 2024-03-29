import torch
import os.path as osp
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint 
from pytorch_lightning.loggers import TensorBoardLogger
from baseline import Baseline
from config import CONFIG
from utils.utils import build_model_name

"""
    Modify training configurations in config.py
"""

# get logger
logger = TensorBoardLogger(save_dir=CONFIG.METADATA.LOG_PATH)

# initialize baselinemodel
model = Baseline()

model_name = build_model_name()
print(model_name)

# save checkpoint every 5 epochs
model_checkpoint = ModelCheckpoint(every_n_epochs=5)
early_stopping = EarlyStopping(monitor='epoch_loss', patience=20, mode='min')

trainer = Trainer(
    accelerator='gpu',
    max_epochs=CONFIG.TRAIN.MAX_EPOCH,
    callbacks=[model_checkpoint, early_stopping],
    logger=logger,
    log_every_n_steps=1,
)
# if trained with checkpoint
if CONFIG.TRAIN.RESUME is not None:
    ckpt_path=CONFIG.TRAIN.RESUME
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    trainer.fit(model=model)
else:
    trainer.fit(model=model)

#save model state dict
torch.save(model.state_dict(), osp.join(CONFIG.METADATA.SAVE_PATH, model_name))


