import math
from neuralmagicML.pytorch.recal import ScheduledModifierManager, ScheduledOptimizer
from neuralmagicML.pytorch.utils import TensorBoardLogger, PythonLogger

manager = ScheduledModifierManager.from_yaml("/PATH/TO/config.yaml")
optimizer = ScheduledOptimizer(
    optimizer,
    MODEL,
    manager,
    steps_per_epoch=math.ceil(len(TRAIN_DATASET) / TRAIN_BATCH_SIZE),
    loggers=[TensorBoardLogger(), PythonLogger()],
)
