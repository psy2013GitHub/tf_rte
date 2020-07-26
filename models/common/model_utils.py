
'''
    ref: https://github.com/zihangdai/xlnet/blob/master/model_utils.py
'''

import re
import tensorflow as tf
from ..common.optimizer import AdamWeightDecayOptimizer

__all__ = ['get_train_op', ]

class _flags:

    def __init__(self, params):
        self.params = params

    def __getattr__(self, item, default=None):
        return self.params.get(item, default)

def get_train_op(params, total_loss, grads_and_vars=None):
  global_step = tf.train.get_or_create_global_step()

  FLAGS = _flags(params)

  # increase the learning rate linearly
  if FLAGS.warmup_steps > 0:
    warmup_lr = (tf.cast(global_step, tf.float32)
                 / tf.cast(FLAGS.warmup_steps, tf.float32)
                 * FLAGS.learning_rate)
  else:
    warmup_lr = FLAGS.learning_rate

  # decay the learning rate
  if FLAGS.decay_method == "poly":
    decay_lr = tf.train.polynomial_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        end_learning_rate=FLAGS.learning_rate * FLAGS.min_lr_ratio)
  elif FLAGS.decay_method == "cos":
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)
  else:
      decay_lr = FLAGS.learning_rate

  learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                           warmup_lr, decay_lr)

  if (FLAGS.weight_decay > 0 and not FLAGS.use_tpu and
      FLAGS.num_core_per_host > 1):
    raise ValueError("Do not support `weight_decay > 0` with multi-gpu "
                     "training so far.")

  if FLAGS.weight_decay == 0:
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        epsilon=FLAGS.adam_epsilon)
  else:
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        epsilon=FLAGS.adam_epsilon,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        weight_decay_rate=FLAGS.weight_decay)

  if FLAGS.use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  if grads_and_vars is None:
    grads_and_vars = optimizer.compute_gradients(total_loss)
  gradients, variables = zip(*grads_and_vars)
  clipped, gnorm = tf.clip_by_global_norm(gradients, FLAGS.clip)

  if getattr(FLAGS, "lr_layer_decay_rate", 1.0) != 1.0:
    n_layer = 0
    for i in range(len(clipped)):
      m = re.search(r"model/transformer/layer_(\d+?)/", variables[i].name)
      if not m: continue
      n_layer = max(n_layer, int(m.group(1)) + 1)

    for i in range(len(clipped)):
      for l in range(n_layer):
        if "model/transformer/layer_{}/".format(l) in variables[i].name:
          abs_rate = FLAGS.lr_layer_decay_rate ** (n_layer - 1 - l)
          clipped[i] *= abs_rate
          tf.logging.info("Apply mult {:.4f} to layer-{} grad of {}".format(
              abs_rate, l, variables[i].name))
          break

  train_op = optimizer.apply_gradients(
      zip(clipped, variables), global_step=global_step)

  # Manually increment `global_step` for AdamWeightDecayOptimizer
  if FLAGS.weight_decay > 0:
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

  return train_op, learning_rate, gnorm