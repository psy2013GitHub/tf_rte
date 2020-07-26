'''
    作者：Celine
    链接：https: // zhuanlan.zhihu.com / p / 106400162
    来源：知乎
    著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''

import numpy as np
import tensorflow as tf

'''
class SessionRunHook(object):
    """Hook to extend calls to MonitoredSession.run()."""

    def begin(self):
        """在创建会话之前调用
        调用begin()时，default graph会被创建，
        可在此处向default graph增加新op,begin()调用后，default graph不能再被修改
        """
        pass

    def after_create_session(self, session, coord):  # pylint: disable=unused-argument
        """tf.Session被创建后调用
        调用后会指示所有的Hooks有一个新的会话被创建
        Args:
          session: A TensorFlow Session that has been created.
          coord: A Coordinator object which keeps track of all threads.
        """
        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """调用在每个sess.run()执行之前
        可以返回一个tf.train.SessRunArgs(op/tensor),在即将运行的会话中加入这些op/tensor；
        加入的op/tensor会和sess.run()中已定义的op/tensor合并，然后一起执行；
        Args:
          run_context: A `SessionRunContext` object.
        Returns:
          None or a `SessionRunArgs` object.
        """
        return None

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):  # pylint: disable=unused-argument
        """调用在每个sess.run()之后
        参数run_values是befor_run()中要求的op/tensor的返回值；
        可以调用run_context.qeruest_stop()用于停止迭代
        sess.run抛出任何异常after_run不会被调用
        Args:
          run_context: A `SessionRunContext` object.
          run_values: A SessionRunValues object.
        """
        pass

    def end(self, session):  # pylint: disable=unused-argument
        """在会话结束时调用
        end()常被用于Hook想要执行最后的操作，如保存最后一个checkpoint
        如果sess.run()抛出除了代表迭代结束的OutOfRange/StopIteration异常外，
        end()不会被调用
        Args:
          session: A TensorFlow Session that will be soon closed.
        """
        pass
'''

class NanTensorHook(tf.train.SessionRunHook):
    """Monitors the loss tensor and stops training if loss is NaN.
    Can either fail with exception or just stop training.
    """

    def __init__(self, learning_rate, fail_on_nan_loss=True):
        """Initializes a `NanTensorHook`.
        Args:
          loss_tensor: `Tensor`, the loss tensor.
          fail_on_nan_loss: `bool`, whether to raise exception when loss is NaN.
        """
        self._learning_rate = learning_rate
        self._fail_on_nan_loss = fail_on_nan_loss


    def begin(self):
        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._learning_rate_tensor = tf.get_default_graph().get_tensor_by_name(
            'learning_rate:0')  # 注意，这里根据name来索引tensor，所以请在定义学习速率的时候，为op添加名字
        self._lrn_rate = 0.1  # 第一阶段的学习速率


    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs(self._loss_tensor)

    def after_run(self, run_context, run_values):
        if np.isnan(run_values.results):
          failure_message = "Model diverged with loss = NaN."
          if self._fail_on_nan_loss:
            logging.error(failure_message)
            raise NanLossDuringTrainingError
          else:
            logging.warning(failure_message)
            # We don't raise an error but we request stop without an exception.
            run_context.request_stop()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            self._global_step_tensor,  # Asks for global step value.
            feed_dict={self._lrn_rate_tensor: self._lrn_rate}
        )  # Sets learning rate

    def after_run(self, run_context: tf.train.SessionRunContext, run_values):
        # tf.estimator
        train_step = run_values.results
        if train_step < 10000:
            pass
        elif train_step < 20000:
            self._lrn_rate = 0.01 # 第二阶段的学习速率
        elif train_step < 30000:
            self._lrn_rate = 0.001 # 第三阶段的学习速率
        else:
            self._lrn_rate = 0.0001 # 第四阶段的学习速率