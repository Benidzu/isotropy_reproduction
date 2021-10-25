import numpy as np
from IPython.display import clear_output
from scipy import cluster as clst
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import math as mt
import scipy as sc


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


def flatten_pooling(inp_representations, representation_dev):
    """ calculating sentence representations by concatenating token representations."""
    max_pad = representation_dev[0].shape[0]
    emb_len = representation_dev[0].shape[1]
    sent_representations = np.zeros(
        (len(representation_dev), max_pad*emb_len), dtype=np.float32)
    for i in range(len(representation_dev)):
        sent_representations[i, :] = inp_representations[i *
                                                         max_pad:i*max_pad+max_pad].flatten()
    return sent_representations


def mean_pooling(inp_representations, representation_dev):
    """ calculating sentence representations by averaging over the tokens."""

    sum_index = 0
    sent_representations = []
    for i in range(len(representation_dev)):
        # sanity check
        if len(representation_dev[i]) == 0:
            sent_representations.append(
                np.zeros(inp_representations[0].shape[0], dtype=np.float32))
            continue
        sent_representations.append(np.mean(
            inp_representations[sum_index: sum_index + (len(representation_dev[i]))], axis=0))
        sum_index = sum_index + len(representation_dev[i])

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
    words = []
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            words.append(sentences[i][j])

    return words
