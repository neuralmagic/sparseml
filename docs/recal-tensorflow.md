## Recalibrating in TensorFlow

The recalibration tooling for TensorFlow is located under `neuralmagicML.tensorflow.recal`.
Inside are APIs designed to make model recalibration as easy as possible.
Additionally, the tooling is designed to work with the previously described config files.

The `ScheduledModifierManager` is used to create modifiers from a config file.
Specifically, `ScheduledModifierManager.from_yaml(/PATH/TO/config.yaml)` should be used.
This will return a new instance of `ScheduledModifierManager` containing the modifiers described in the config file.
Once a manager class has been created, it can be used to modify the training graph.
Under the scope of a TensorFlow graph containing the created model operations and variables,
`manager.create_ops()` must be called.
This will go through and modify the graph as defined in the config.yaml file.
It will return modifying ops as a list and modifying extras as a dictionary.
The list of ops must be run once per training step and after the optimizer has run.
Also, `create_ops()` must be called outside of a session to have any effect.
Under the scope of a created TensorFlow session, `manager.initialize_session()` should be called
if the global variables initializer was not run.
This will handle initializing any variables that were created as part of the `create_ops()` invocation.
Finally, after training is complete, `manager.complete_graph()` must be called.
This will clean up the modified graph properly for saving and export.

Example:
```python
import math
from tensorflow.examples.tutorials.mnist import input_data
from neuralmagicML.tensorflow.utils import tf_compat, batch_cross_entropy_loss
from neuralmagicML.tensorflow.models import mnist_net
from neuralmagicML.tensorflow.recal import ScheduledModifierManager

dataset = input_data.read_data_sets("./mnist_data", one_hot=True)
batch_size = 1024
num_batches = int(len(dataset.train.images) / batch_size)

with tf_compat.Graph().as_default() as graph:
    inputs = tf_compat.placeholder(
        tf_compat.float32, [None, 28, 28, 1], name="inputs"
    )
    labels = tf_compat.placeholder(tf_compat.float32, [None, 10])
    logits = mnist_net(inputs)
    loss = batch_cross_entropy_loss(logits, labels)

    global_step = tf_compat.train.get_or_create_global_step()
    manager = ScheduledModifierManager.from_yaml("/PATH/TO/config.yaml")
    mod_ops, mod_extras = manager.create_ops(steps_per_epoch=math.ceil(len(dataset.train.images) / batch_size))
    
    train_op = tf_compat.train.AdamOptimizer(learning_rate=1e-4).minimize(
        loss, global_step=global_step
    )
    
    with tf_compat.Session() as sess:
        sess.run(tf_compat.global_variables_initializer())
        
        for epoch in range(manager.max_epochs):
            for batch in range(num_batches):
                batch_xs, batch_ys = dataset.train.next_batch(batch_size)
                batch_xs = batch_xs.reshape([-1, 28, 28, 1])
                sess.run(train_op, feed_dict={inputs: batch_xs, labels: batch_ys})
                sess.run(mod_ops)
```

Note: if you would like to log to TensorBoard, summaries for the modifying ops and variables are created by default.
They are added to the default TensorFlow summaries collection, and they are additionally made 
accessible under the mod_extras returned from the `manager.create_ops()` function.
Example:
```python
from neuralmagicML.tensorflow.recal import EXTRAS_KEY_SUMMARIES

summary_ops = mod_extras[EXTRAS_KEY_SUMMARIES]
```

Note: if you are using any learning rate modifiers, then the learning rate tensor will need to be passed
from the `manager.create_ops()` function to your optimizer.
Example:
```python
from neuralmagicML.tensorflow.recal import EXTRAS_KEY_LEARNING_RATE

learning_rate = mod_extras[EXTRAS_KEY_LEARNING_RATE]
train_op = tf_compat.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
    loss, global_step=global_step
)
```

Note: if you are using any trainable params modifiers, then the trainable vars in TensorFlow must be collected
and passed to your optimizer in place of the default.
```python
train_op = tf_compat.train.AdamOptimizer(learning_rate=1e-4).minimize(
    loss, global_step=global_step, var_list=tf_compat.trainable_variables()
)
```