
import os
import random
import sys
from pathlib import Path
import tensorflow as tf
import functools
from collections import Counter
import numpy as np
from .bert_formatter import convert_single_instance

MINCOUNT = 1

def read_snli_line(s):
    segs = s.strip().split('\t')
    gold_label, \
    sentence1_binary_parse, sentence2_binary_parse, \
    sentence1_parse, sentence2_parse, \
    sentence1, sentence2, \
    captionID, pairID = segs[:9]
    label15 = segs[9:]
    return gold_label, \
        sentence1_binary_parse, sentence2_binary_parse, \
        sentence1_parse, sentence2_parse, \
        sentence1, sentence2, \
        captionID, pairID, label15

def parse_fn(line, encode=True, with_char=False, bert_out=False, bert_proj_path=None, bert_config_json=None, max_seq_len=512):
    # Encode in Bytes for TF
    label, \
    sentence1_binary_parse, sentence2_binary_parse, \
    sentence1_parse, sentence2_parse, \
    sentence1, sentence2, \
    captionID, pair_id, label15 = read_snli_line(line)
    premise_words = ['<SOS>']
    premise_words.extend([w.encode() if encode and not bert_out else w for w in sentence1_binary_parse.split() if not w in ('(', ')')])
    premise_words.append('<EOS>')

    hypothesis_words = ['<SOS>']
    hypothesis_words.extend([w.encode() if encode and not bert_out else w for w in sentence2_binary_parse.split() if not w in ('(', ')')])
    premise_words.append('<EOS>')

    n_premise_words, n_hypothesis_words = len(premise_words), len(hypothesis_words)
    if bert_out:
        assert bert_proj_path, 'bert_proj_path must not be None'
        sys.path.append(os.path.expanduser(bert_proj_path))
        from models.bert.tokenization import FullTokenizer
        tokenizer = FullTokenizer(vocab_file=bert_config_json['vocab_file'], do_lower_case=bert_config_json['do_lower_case'])
        premise_input_ids, input_mask, segment_ids, n_words = convert_single_instance(premise_words, max_seq_len, tokenizer)
        words = premise_input_ids, input_mask, segment_ids

    pair_id = pair_id.encode() if encode else pair_id
    label = label.encode() if encode else label
    if not with_char:
        return ((premise_words, n_premise_words), (hypothesis_words, n_hypothesis_words), pair_id), label
    else:
        # Chars
        # lengths = [len(c) for c in chars]
        # max_len = max(lengths)
        # chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
        raise NotImplementedError('with_char=True error')
        return ((words, n_words), (chars, lengths)), tags

def generator_fn(fname, encode=True, with_char=False, bert_out=False, bert_proj_path=None, bert_config_json=None, max_seq_len=512):
    with Path(fname).expanduser().open('r') as fid:
        next(fid) # 跳过首行
        for line in fid:
            _, label = parse_fn(line, encode=encode, with_char=with_char,
                               bert_out=bert_out, bert_proj_path=bert_proj_path, bert_config_json=bert_config_json, max_seq_len=max_seq_len)
            label_str = label.decode().strip() if encode else label.strip()
            if label_str == '-':
                continue
            # print('{}\ {}'.format(label, line))
            yield _, label

def input_fn(file, params=None, shuffle_and_repeat=False, with_char=False, bert_out=False,
             bert_proj_path=None, bert_config_json=None, max_seq_len=512):
    params = params if params is not None else {}
    if bert_out:
        if not with_char:
            shapes = ((([None], [None], [None]), ()), [None])
            types = (((tf.int32, tf.int32, tf.int32), tf.int32), tf.string)
            defaults = (((0, 0, 0), 0), 'O')
        else:
            shapes = (
                (
                    (([None], [None], [None]), ()),
                    ([None, None], [None])
                ),  # (chars, nchars)
                [None]
            )  # tags
            types = (
                (
                    ((tf.int32, tf.int32, tf.int32), tf.int32),
                    (tf.string, tf.int32)
                ),
                tf.string
            )
            defaults = (
                (
                    ((0, 0, 0), 0),
                    ('<pad>', 0)
                ),
                'O'
            )
    else:
        if not with_char:
            shapes = (([None], ()), ([None], ()), ()), ()
            types = ((tf.string, tf.int32), (tf.string, tf.int32), tf.string), tf.string
            defaults = (('<pad>', 0), ('<pad>', 0), 'empty'), 'neutral'
        else:
            shapes = (
                (
                    ([None], ()),  # (words, nwords)
                    ([None, None], [None])
                ),  # (chars, nchars)
                [None]
            )  # tags
            types = (
                (
                    (tf.string, tf.int32),
                    (tf.string, tf.int32)
                ),
                tf.string
            )
            defaults = (
                (
                    ('<pad>', 0),
                    ('<pad>', 0)
                ),
                'O'
            )

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, file, with_char=with_char, bert_out=bert_out,
                          bert_proj_path=bert_proj_path, bert_config_json=bert_config_json, max_seq_len=max_seq_len),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset

def build_vocab(files, output_dir, min_count=MINCOUNT, force_build=False, encode=False, top_freq_words=None):
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save

    words_path = '{}/vocab.words.txt'.format(output_dir)
    chars_path = '{}/vocab.chars.txt'.format(output_dir)
    tags_path = '{}/vocab.tags.txt'.format(output_dir)

    if not force_build:
        if Path(words_path).expanduser().exists() \
            and Path(chars_path).expanduser().exists() \
            and Path(tags_path).expanduser().exists():
            print('vocab already build, pass. {} {} {}'.format(words_path, chars_path, tags_path))
            return words_path, chars_path, tags_path

    print('Build vocab words/tags (may take a while)')
    counter_words = Counter()
    vocab_labels = set()
    for file in files:
        for _ in generator_fn(file, encode=encode):
            ((premise_words, n_premise_words), (hypothesis_words, n_hypothesis_words), pair_ids), label = _
            counter_words.update(premise_words)
            counter_words.update(hypothesis_words)
            vocab_labels.add(label)

    vocab_words = {w for i, (w, c) in enumerate(counter_words.most_common(top_freq_words if top_freq_words and top_freq_words > 0 else None)) if c >= min_count}

    with Path(words_path).expanduser().open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path(chars_path).expanduser().open('w') as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))


    with Path(tags_path).expanduser().open('w') as f:
        for t in sorted(list(vocab_labels)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_labels)))

    return words_path, chars_path, tags_path


def build_glove(words_file='vocab.words.txt', output_path='glove.npz', glove_path='glove.840B.300d.txt', force_build=False):

    if not force_build:
        if Path(output_path).expanduser().exists():
            print('glove already build, pass. {}'.format(output_path))
            return output_path

    with Path(words_file).expanduser().open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.random.randn(size_vocab, 300) * 0.01

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with Path(glove_path).expanduser().open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed(str(Path(output_path).expanduser()), embeddings=embeddings)

    return output_path
