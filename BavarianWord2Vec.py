import bz2
import collections
from ftplib import FTP
from io import BytesIO, StringIO
import math
import random
import re
from typing import Callable, List, Tuple

from nltk import sent_tokenize, word_tokenize
import numpy as np
import tensorflow as tf
from wikiextractor import WikiExtractor

batch_size = 128
embedding_size = 128
window_size = 4
valid_size = 16
valid_window = 50
min_presence_in_corpus = 15
num_steps = 100000

valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)

neg_samples = 32


def get_and_preprocess_text_from_wikipedia(mirror: str, folder: str, filename: str) -> List[str]:
    ftp = FTP(mirror)
    ftp.login()
    ftp.cwd(folder)

    file = BytesIO()
    ftp.retrbinary(filename, file.write)
    ftp.quit()
    file.seek(0)
    unedited_wikipedia_text = bz2.decompress(file.getvalue()).split(b'\n')
    file.close()

    preprocessed_text = []
    for page_data in WikiExtractor.pages_from(unedited_wikipedia_text):
        id, revid, title, ns, catSet, page = page_data
        if WikiExtractor.keepPage(ns, catSet, page):
            tokenized_text = []
            out = StringIO()
            try:
                e = WikiExtractor.Extractor(id, revid, title, page)
                e.extract(out)
                extracted_text = out.getvalue()
                extracted_text = re.sub('\<doc.*\>', '', extracted_text)
                extracted_text = re.sub('([0-9]+([\.\,]))*[0-9]+', 'NUM', extracted_text)
                extracted_text = re.sub('.*\n\n', '', extracted_text, count=1)
                extracted_text = extracted_text.split('</doc>')
                for article in extracted_text:
                    article = sent_tokenize(article)
                    for sentence in article:
                        if '===' in sentence:
                            print("Unpreprocessed")
                            print(sentence)
                        sentence = re.sub(r'[^\w\s]', '', sentence, re.UNICODE)
                        if '===' in sentence:
                            print("Preprocessed")
                            print(sentence)
                        sentence = word_tokenize(sentence)
                        tokenized_text.extend(sentence)
            except error as error:
                print(error)
            out.truncate(0)
            out.seek(0)
            preprocessed_text.extend(tokenized_text)
    return preprocessed_text


def build_dataset(words: List[str], minimum_count_threshold: int) -> Tuple[List[int],
                                                                           List[Tuple[str, int]],
                                                                           dict, dict, int]:
    count = [['UNK', -1]]
    counts = collections.Counter(words).most_common()
    counts = [value_list for value_list in counts if value_list[1] >= minimum_count_threshold]
    count.extend(counts)
    vocabulary_size = len(count)
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0

    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count = unk_count + 1
        data.append(index)

    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    assert len(dictionary) == vocabulary_size
    return data, count, dictionary, reverse_dictionary, vocabulary_size


def generate_batch_skip_gram(batch_size: int, window_size: int) -> Tuple[Callable[[np.ndarray], np.array],
                                                                         Callable[[np.ndarray], np.array]]:
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * window_size + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    num_samples = 2 * window_size

    for i in range(batch_size // num_samples):
        k = 0
        for j in list(range(window_size)) + list(range(window_size + 1, 2 * window_size + 1)):
            batch[i * num_samples + k] = buffer[window_size]
            labels[i * num_samples + k, 0] = buffer[j]
            k += 1

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


preprocessed_text = get_and_preprocess_text_from_wikipedia('ftpmirror.your.org',
                                                           'pub/wikimedia/dumps/barwiki/20200801/',
                                                           "RETR barwiki-20200801-pages-meta-current.xml.bz2")
data, count, dictionary, reverse_dictionary, vocabulary_size = build_dataset(preprocessed_text, min_presence_in_corpus)

data_index = 0
tf.reset_default_graph()

train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=0.5 / math.sqrt(embedding_size))
)
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))
embed = tf.nn.embedding_lookup(embeddings, train_dataset)

loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(
        weights=softmax_weights, biases=softmax_biases, inputs=embed,
        labels=train_labels, num_sampled=neg_samples, num_classes=vocabulary_size)
)
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

skip_losses = []
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    tf.global_variables_initializer().run()
    average_loss = 0

    for step in range(num_steps):

        batch_data, batch_labels = generate_batch_skip_gram(
            batch_size, window_size)

        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += l

        if (step + 1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000

            skip_losses.append(average_loss)
            print('Average loss at step %d: %f' % (step + 1, average_loss))
            average_loss = 0

        # Evaluating validation set word similarities
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    skip_gram_final_embeddings = normalized_embeddings.eval()

np.save('skip_embeddings', skip_gram_final_embeddings)
