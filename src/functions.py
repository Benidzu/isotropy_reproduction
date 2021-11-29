import numpy as np
from IPython.display import clear_output
from scipy import cluster as clst
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import math as mt
import scipy as sc
from copy import deepcopy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef


def sper_corrcoef(targets, predictions):
    """Spearman correlation coefficient."""
    return 100 * sc.stats.spearmanr(targets, predictions)[0]


def get_model_features(df, batch_size, encodings, model, embarray):
    """ Send sentence token encodings through model and return ndarray of embeddings for each sentence
    and each token."""
    i = 0
    while i < len(df):
        lower = i
        upper = min(i+batch_size, len(df))
        output = model(np.asarray(encodings["input_ids"][lower:upper], dtype="int32"),
                       attention_mask=np.asarray(
                           encodings["attention_mask"][lower:upper], dtype="int32"),
                       token_type_ids=np.asarray(encodings["token_type_ids"][lower:upper], dtype="int32"))[0]

        embarray[lower:upper, :, :] = np.asarray(output)
        i += batch_size
        print(i)
        if i % 100 == 0:
            clear_output()

    return embarray


def flatten_pooling(word_representations, sentences_full_rep):
    """ calculating sentence representations by concatenating token representations."""
    max_pad = sentences_full_rep[0].shape[0]
    emb_len = sentences_full_rep[0].shape[1]
    sent_representations = np.zeros(
        (len(sentences_full_rep), max_pad*emb_len), dtype=np.float32)
    for i in range(len(sentences_full_rep)):
        sent_representations[i, :] = word_representations[i *
                                                          max_pad:i*max_pad+max_pad].flatten()
    return sent_representations


def mean_pooling(word_representations, sentences_full_rep):
    """ calculating sentence representations by averaging over the tokens."""

    sum_index = 0
    sent_representations = []
    for i in range(len(sentences_full_rep)):
        # sanity check
        if len(sentences_full_rep[i]) == 0:
            sent_representations.append(
                np.zeros(word_representations[0].shape[0], dtype=np.float32))
            continue
        sent_representations.append(np.mean(
            word_representations[sum_index: sum_index + (len(sentences_full_rep[i]))], axis=0))
        sum_index = sum_index + len(sentences_full_rep[i])

    return sent_representations


def similarity(sent_rep, emb_length):
    """ calculating cosine similarity between two sentences."""

    score = []
    l = 0
    for i in range(int(len(sent_rep)/2)):
        score.append(cosine_similarity(np.reshape(sent_rep[l], (1, emb_length)),
                                       np.reshape(sent_rep[l + 1], (1, emb_length)))[0][0])
        l = l + 2

    return score


def isotropy(representations):
    """Calculating isotropy of embedding space based on Eq.2
           arg:
              representations (n_samples, n_dimensions)
            """

    eig_values, eig_vectors = np.linalg.eig(np.matmul(np.transpose(representations),
                                                      representations))
    max_f = -mt.inf
    min_f = mt.inf

    # eigvectors are in columns
    for i in range(eig_vectors.shape[1]):
        f = np.matmul(representations, np.expand_dims(eig_vectors[:, i], 1))
        f = np.sum(np.exp(f))

        min_f = min(min_f, f)
        max_f = max(max_f, f)

    isotropy = min_f / max_f

    return isotropy


def cluster_based(representations, n_cluster: int, n_pc: int, emb_length):
    """ Improving Isotropy of input representations using cluster-based method
        Args: 
              inputs:
                    representations: 
                      input representations numpy array(n_samples, n_dimension)
                    n_cluster: 
                      the number of clusters
                    n_pc: 
                      the number of directions to be discarded
              output:
                    isotropic representations (n_samples, n_dimension)

              """
    label = []
    if n_cluster != 1:
        centroid, label = clst.vq.kmeans2(representations, n_cluster, minit='points',
                                          missing='warn', check_finite=True)
    else:
        label = np.zeros(len(representations), dtype=np.int32)
    # calculate cluster mean embedding
    cluster_mean = []
    for i in range(max(label)+1):
        sum = np.zeros([1, emb_length])
        for j in np.nonzero(label == i)[0]:
            sum = np.add(sum, representations[j])
        cluster_mean.append(sum/len(label[label == i]))

    zero_mean_representation = []
    for i in range(len(representations)):
        zero_mean_representation.append(
            (representations[i])-cluster_mean[label[i]])

    cluster_representations = {}
    for i in range(n_cluster):
        cluster_representations.update({i: {}})
        for j in range(len(representations)):
            if (label[j] == i):
                cluster_representations[i].update(
                    {j: zero_mean_representation[j]})

    cluster_representations2 = []
    for j in range(n_cluster):
        cluster_representations2.append([])
        for key, value in cluster_representations[j].items():
            cluster_representations2[j].append(value)

    cluster_representations2 = np.array(cluster_representations2)

    print("start PCA")
    model = PCA(svd_solver="randomized")
    post_rep = np.zeros((representations.shape[0], representations.shape[1]))

    # Can return errors if n_cluster > number of datapoints in smallest cluster. No issues if empty cluster however.
    for i in range(n_cluster):
        model.fit(np.array(cluster_representations2[i]).reshape(
            (-1, emb_length)))
        component = np.reshape(model.components_, (-1, emb_length))
        print(i, "; ",)
        for index in cluster_representations[i]:
            sum_vec = np.zeros((1, emb_length))

            for j in range(n_pc):
                sum_vec = sum_vec + np.dot(cluster_representations[i][index],
                                           np.transpose(component)[:, j].reshape((emb_length, 1))) * component[j]

            post_rep[index] = cluster_representations[i][index] - sum_vec

    clear_output()

    return post_rep


def get_best_classifier(epochs, X_tr, Y_tr, X_dev, Y_dev, scorer="", random_state=None):
    """ Get the best MLP classifier based on validation set score in a specific epoch.
        Inputs:
            epochs: number of epochs to train for
            X_tr: training set input - numpy array of size (n,m)
            Y_tr: training set labels - numpy array of size (n,)
            X_dev: validation set input - numpy array of size (k,m)
            Y_dev: validation set labels - numpy array of size (k,)
        Outputs:
            clf_best: best MLP classifier
            scores: all epochs MLP classifier scores on validation set.
    """
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, verbose=False,
                        early_stopping=False, activation="relu", solver="adam", learning_rate_init=5e-3, random_state=random_state)
    clf_best = None
    score_best = -1
    scores = []
    for i in range(epochs):
        clf.partial_fit(X_tr, Y_tr, classes=[0, 1])
        score = -1
        if scorer == "matthew":
            score = matthews_corrcoef(Y_dev, clf.predict(X_dev))
        else:
            score = clf.score(X_dev, Y_dev)
        print("epoch " + str(i+1) + ", score: " + str(score))
        scores.append(score)
        if score > score_best:
            score_best = score
            clf_best = deepcopy(clf)

    return clf_best, scores


def global_method(representations, n_pc: int, emb_length):
    """ Improving Isotropy of input representations using cluster-based method
        Args: 
              inputs:
                    representations: 
                      input representations numpy array(n_samples, n_dimension)
                    n_pc: 
                      the number of directions to be discarded
              output:
                    improved representations using global method (n_samples, n_dimension)
              """

    post_rep = cluster_based(representations, 1, n_pc, emb_length)
    return post_rep


def get_representations(data_, tokenizer, model, emb_length):
    """ Get sentence full representations for STS data.
        Args:
            inputs:
                data_: dataframe with raw sentences in 'sentence1' and 'sentence2' columns
                tokenizer: model tokenizer from transformers library
                model: model from transformers library
                emb_length: dimensionality of single token embedding of used model (output layer size)
            output:
                sentences: list of sentences embeddings (2*len(data_), num_tokens_in_sentnce, emb_length)
            """
    sentences = []
    for i in range(len(data_)):
        print(i)
        # First sentence
        inputs = tokenizer.encode(
            data_['sentence1'].iloc[i], add_special_tokens=True)
        inputs = np.asarray(inputs, dtype='int32').reshape((1, -1))

        # getting the representation of the last layer
        output = model(inputs)[0]
        output = np.asarray(output).reshape((-1, emb_length))

        # Removing CLS and SEP tokens
        idx = [0, len(output)-1]
        output = np.delete(output, idx, axis=0)
        output = np.asarray(output).reshape((-1, emb_length))

        sentences.append(output)

        # Second sentence
        inputs = tokenizer.encode(
            data_['sentence2'].iloc[i], add_special_tokens=True)
        inputs = np.asarray(inputs, dtype='int32').reshape((1, -1))

        output = model(inputs)[0]
        output = np.asarray(output).reshape((-1, emb_length))

        # Removing CLS and SEP tokens
        idx = [0, len(output)-1]
        output = np.delete(output, idx, axis=0)
        output = np.asarray(output).reshape((-1, emb_length))

        sentences.append(output)
        if i % 10 == 0:
            clear_output()

    return sentences


def getWords(sentences):
    """ Get words (tokens) representations in a list by removing the sentences axis. """
    words = []
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            words.append(sentences[i][j])

    return words
