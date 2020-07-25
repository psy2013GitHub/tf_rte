"""
Enhanced LSTM for Natural Language Inference
https://arxiv.org/pdf/1609.06038v3.pdf
"""

__author__ = ""

import json
from models.common.log_utils import init_log_path
import shutil

from tensorflow.python.keras import backend as K
from tensorflow.python import array_ops
from tensorflow.python import debug as tf_debug
from tf_metrics import precision, recall, f1
from models.common.embedding_layer import embedding_layer

# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

from data.snli_input import *

# Logging

def bi_lstm_encoder(x, seq_len, lstm_size, scope_name='lstm_encoder', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope_name, reuse=reuse):
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(lstm_size, reuse=reuse)
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(lstm_size, reuse=reuse)
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)

        t = tf.transpose(x, perm=[1, 0, 2])
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=seq_len) # 出于效率考虑，tf仅处理当前batch最大长度
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=seq_len)

        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.transpose(output, perm=[1, 0, 2])

        # output = K.math_ops.multiply(output, tf.cast(tf.sequence_mask(seq_len, tf.reduce_max(seq_len)), output.dtype))

        return output

def add_mask(x, mask, expand_axis=None):
    '''
        对x进行mask
    :param x:
    :param mask:
    :param mode:
    :return:
    '''
    if mask is None:
        return x
    if expand_axis:
        for _ in expand_axis:
            mask = array_ops.expand_dims(mask, _)
    return x - (1.0 - tf.cast(mask, x.dtype)) * 1e10

def pairwise_attention_dot(x1, x2, x1_mask=None, x2_mask=None, return_score=False, scope_name='pairwise_attention_dot', reuse=tf.AUTO_REUSE):
    '''

    :param x1: [N, S1, d]
    :param x2: [N, S2, d]
    :param x1_mask: [N, S1]
    :param x2_mask: [N, S2], 1 as valid position
    :return:
    '''
    with tf.variable_scope(scope_name, reuse=reuse):
        alpha = tf.matmul(x1, tf.transpose(x2, perm=[0, 2, 1])) # [N, S1, S2]
        alpha1 = alpha
        if not x2_mask is None:
            alpha1 = add_mask(alpha, x2_mask, expand_axis=(1,))

        alpha2 = alpha
        if not x1_mask is None:
            alpha2 = add_mask(alpha, x1_mask, expand_axis=(2,))

        alpha1 = tf.nn.softmax(alpha1, axis=2)
        alpha2 = tf.nn.softmax(alpha2, axis=1)
        x1_att = K.batch_dot(alpha1, x2, axes=[2, 1])
        x2_att = K.batch_dot(alpha2, x1, axes=[1, 1])
        if return_score:
            return x1_att, x2_att, alpha1, alpha2, alpha
        return x1_att, x2_att

def seq_reduce_mean(x, mask):
    mask = tf.cast(mask, x.dtype)
    mask = tf.expand_dims(mask, -1)
    return tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)

def seq_reduce_max(x, mask):
    mask = tf.cast(mask, x.dtype)
    mask = tf.expand_dims(mask, -1)
    mask = mask + (1 - mask) * 1e-12
    return tf.reduce_max(x * mask, axis=1)

def set_pad_zero(x, seq_len):
    mask = tf.sequence_mask(seq_len, tf.shape(x)[1])
    return K.math_ops.multiply(x, tf.cast(mask, x.dtype))

def model_fn(features, labels, mode, params):
    if isinstance(features, dict):
        features = features['words'], features['nwords']

    # Read vocabs and inputs
    dropout = params['dropout']
    (premise_words, n_premise_words), (hypothesis_words, n_hypothesis_words), pair_ids = features
    premise_mask = tf.sequence_mask(n_premise_words, maxlen=tf.reduce_max(n_premise_words))
    hypothesis_mask = tf.sequence_mask(n_hypothesis_words, maxlen=tf.reduce_max(n_hypothesis_words))

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # 将label转化成int
    with Path(params['labels']).open() as f:
        indices = [idx for idx, label in enumerate(f) if label.strip() != 'neutral']
        num_classes = len(indices) + 1
    vocab_labels = tf.contrib.lookup.index_table_from_file(params['labels'])
    if not labels is None: # train/eval
        labels = vocab_labels.lookup(labels, name='labels_lookup')
        print('labels: {}'.format(labels))

    # 将词转化为int
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])

    with Path(params['words']).open() as f:
        words_vocab_size = len(f.readlines()) + params['num_oov_buckets']

    # Word Embeddings
    premise_word_ids = vocab_words.lookup(premise_words) + 1
    premise_word_ids = set_pad_zero(premise_word_ids, n_premise_words)
    hypothesis_word_ids = vocab_words.lookup(hypothesis_words) + 1
    hypothesis_word_ids = set_pad_zero(hypothesis_word_ids, n_premise_words)

    with tf.variable_scope('embedding') as scope:
        if params.get('rand_embedding', False):
            premise_embeddings = embedding_layer(premise_word_ids, words_vocab_size, params['dim'], zero_pad=False, reuse=False)
            hypothesis_embeddings = embedding_layer(hypothesis_word_ids, words_vocab_size, params['dim'], zero_pad=False, reuse=True)
        else:
            # todo 此处词不在glove表格咋办？，比如某些稀有名词
            glove = np.load(str(Path(params['glove']).expanduser()))['embeddings']  # np.array
            variable = np.vstack([[[0.]*params['dim']], glove]) # pad 0
            variable = np.vstack([variable, np.zeros((params['num_oov_buckets'], params['dim']),)]) # num_oov_buckets
            variable = tf.Variable(variable, dtype=tf.float32, trainable=True)
            premise_embeddings = tf.nn.embedding_lookup(variable, premise_word_ids, name='premise_embeddings')
            hypothesis_embeddings = tf.nn.embedding_lookup(variable, hypothesis_word_ids, name='hypothesis_embeddings')

        premise_embeddings = tf.layers.dropout(premise_embeddings, rate=dropout, training=training)
        hypothesis_embeddings = tf.layers.dropout(hypothesis_embeddings, rate=dropout, training=training)

    with tf.variable_scope('encoder') as scope:
        # Bi-LSTM编码，Premise和Hypothesis是用同一个lstm吗？
        premise_x = bi_lstm_encoder(premise_embeddings, n_premise_words, params['lstm_size'], scope_name='lstm-encoder', reuse=False)
        # premise_x = tf.layers.dropout(premise_x, rate=dropout, training=training)

        hypothesis_x = bi_lstm_encoder(hypothesis_embeddings, n_hypothesis_words, params['lstm_size'], scope_name='lstm-encoder', reuse=True)
        # hypothesis_x = tf.layers.dropout(hypothesis_x, rate=dropout, training=training)

    with tf.variable_scope('merge') as scope:
        # attention融合x, f(x|y)层，是在同一向量空间吗？
        premise_attn_x, hypothesis_attn_x = pairwise_attention_dot(premise_x, hypothesis_x, x1_mask=premise_mask, x2_mask=hypothesis_mask, return_score=False)
        premise_x = tf.concat([premise_x, premise_attn_x, premise_x - premise_attn_x, tf.multiply(premise_x, premise_attn_x)], axis=-1)
        hypothesis_x = tf.concat([hypothesis_x, hypothesis_attn_x, hypothesis_x - hypothesis_attn_x, tf.multiply(hypothesis_x, hypothesis_attn_x)], axis=-1)

        premise_x = tf.layers.dense(premise_x, units=params['lstm_size'], activation='relu', name='merge', reuse=False)
        premise_x = tf.layers.dropout(premise_x, rate=dropout, training=training)

        hypothesis_x = tf.layers.dense(hypothesis_x, units=params['lstm_size'], activation='relu', name='merge', reuse=True)
        hypothesis_x = tf.layers.dropout(hypothesis_x, rate=dropout, training=training)

    with tf.variable_scope('decoder') as scope:
        # Bi-LSTM编码 融合层
        premise_x = bi_lstm_encoder(premise_x, n_premise_words, params['lstm_size'], scope_name='lstm-composer', reuse=False)
        # premise_x = tf.layers.dropout(premise_x, rate=dropout, training=training)

        hypothesis_x = bi_lstm_encoder(hypothesis_x, n_hypothesis_words, params['lstm_size'], scope_name='lstm-composer', reuse=True)
        # hypothesis_x = tf.layers.dropout(hypothesis_x, rate=dropout, training=training)

    with tf.variable_scope('pooler') as scope:
        # Pooling层
        x = tf.concat([
            seq_reduce_mean(premise_x, premise_mask),
            seq_reduce_max(premise_x, premise_mask),
            seq_reduce_mean(hypothesis_x, hypothesis_mask),
            seq_reduce_max(hypothesis_x, hypothesis_mask)
        ], axis=-1)
        x = tf.layers.dropout(x, rate=dropout, training=training)

    with tf.variable_scope('classifier') as scope:
        # 分类层
        x = tf.layers.dense(x, params['hidden_dim'], activation='tanh')
        x = tf.layers.dropout(x, rate=dropout, training=training)
        logits = tf.layers.dense(x, num_classes, activation=None)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        pred_ids = tf.argmax(logits, axis=1)

    # loss
    if not labels is None:
        one_hot_labels = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_labels = tf.contrib.lookup.index_to_string_table_from_file(
            params['labels'])
        pred_strings = reverse_vocab_labels.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'labels': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        # Metrics=
        # tf.Print(labels, [labels], message='\nlabels:')
        metrics = {
            'acc': tf.metrics.accuracy(labels, pred_ids),
            'precision': precision(labels, pred_ids, num_classes, indices),
            'recall': recall(labels, pred_ids, num_classes, indices),
            'f1': f1(labels, pred_ids, num_classes, indices),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'top_freq_words': 42394,
        'max_seq_len': 256,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1, # ？
        'epochs': 5000,
        'batch_size': 256,
        'buffer': 15000, # ？
        'lstm_size': 300,
        'hidden_dim': 300,
        'learning_rate': 4e-4,
        'force_build_vocab': True,
        'vocab_dir': './',
        'rand_embedding': False, # 随机初始化embedding
        'force_build_glove': True,
        'glove': './glove.npz',
        'pretrain_glove': '~/.datasets/embeddings/glove.840B.300d/glove.840B.300d.txt',
        'files': [
            '~/.datasets/rte/snli_1.0/snli_1.0_train.txt',
            '~/.datasets/rte/snli_1.0/snli_1.0_dev.txt',
            '~/.datasets/rte/snli_1.0/snli_1.0_test.txt'
        ],
        'DATADIR': '~/.datasets/rte/snli_1.0/',
        'RESULT_DIR': './results/'
    }

    init_log_path(params['RESULT_DIR'])

    with Path('{}/params.json'.format(params['RESULT_DIR'])).open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fname(name):
        return str(Path(params['DATADIR'], 'snli_1.0_{}.txt'.format(name)).expanduser())

    params['words'], params['chars'], params['labels'] = build_vocab(params['files'],
        params['vocab_dir'],
        force_build=params['force_build_vocab'],
        top_freq_words=params['top_freq_words']
    )

    params['glove'] = build_glove(words_file=params['words'],
          output_path=params['glove'],
          glove_path=params['pretrain_glove'],
          force_build=params['force_build_glove']
    )

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fname('train'), params, max_seq_len=params['max_seq_len'], shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fname('dev'), max_seq_len=params['max_seq_len'])

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True
    )

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120, session_config=session_config)
    model_path = '{}/model'.format(params['RESULT_DIR'])
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    estimator = tf.estimator.Estimator(model_fn, model_path, cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hooks = [
        tf.estimator.experimental.stop_if_no_decrease_hook(estimator, 'acc', 50, min_steps=8000, run_every_secs=120),
        # tf_debug.LocalCLIDebugHook(ui_type="readline"),
    ]
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        Path('{}/score'.format(params['RESULT_DIR'])).mkdir(parents=True, exist_ok=True)
        with Path('{}/score/{}.preds.txt'.format(params['RESULT_DIR'], name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fname(name), max_seq_len=params['max_seq_len'])
            golds_gen = generator_fn(fname(name), encode=True)
            # print(next(golds_gen))
            # raise ValueError
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((premise_words, n_premise_words), (hypothesis_words, n_hypothesis_words), pair_id), label = golds
                f.write(b' '.join([pair_id, label, preds['labels']]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'dev', 'test']:
        write_predictions(name)
