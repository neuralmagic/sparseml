import math
from neuralmagicML.tensorflow.utils import tf_compat
from neuralmagicML.tensorflow.recal import (
    ScheduledModifierManager,
    EXTRAS_KEY_SUMMARIES,
    EXTRAS_KEY_LEARNING_RATE,
)


with tf_compat.Graph().as_default() as graph:
    CREATE_MODEL_GRAPH = None

    global_step = tf_compat.train.get_or_create_global_step()
    manager = ScheduledModifierManager.from_yaml("/PATH/TO/config.yaml")
    mod_ops, mod_extras = manager.create_ops(
        steps_per_epoch=math.ceil(len(TRAIN_DATASET) / TRAIN_BATCH_SIZE)
    )
    summary_ops = mod_extras[EXTRAS_KEY_SUMMARIES]
    learning_rate = mod_extras[EXTRAS_KEY_LEARNING_RATE]

    with tf_compat.Session() as sess:
        sess.run(tf_compat.global_variables_initializer())

        for epoch in range(manager.max_epochs):
            for batch in range(TRAIN_BATCH_SIZE):
                sess.run(TRAIN_OP)
                sess.run(mod_ops)
