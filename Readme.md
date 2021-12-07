# [Re] A Cluster-based Approach for Improving Isotropy in Contextual Embedding Space

This repository contains code for the reproduction of the [selected paper](https://aclanthology.org/2021.acl-short.73.pdf) for [ML Reproducibility Challenge 2021](https://paperswithcode.com/rc2021).

The author's originally provided code is available [here](https://github.com/Sara-Rajaee/clusterbased_isotropy_enhancement).

## Environment setup
Tested and working with conda package manager, on Windows 10. First clone repo:

``git clone https://github.com/Benidzu/isotropy_reproduction.git ``

Next create a new conda environment and install all necessary dependencies with:

``conda env create -f environment.yml ``

Alternatively, you could manually install the following packages:

```
matplotlib
jupyter   
numpy
python
pandas
nb_conda
tensorflow
scikit-learn
datasets
nltk
tqdm==4.62.3
transformers==2.11.0
```

## Data

To run the ipython notebooks in ``src/``, you will need to put data in the corresponding sub-folder in ``data/``. 

STS-2012 to STS-2016 and STS-B (STS-benchmark) data is available [here](https://ixa2.si.ehu.eus/stswiki/index.php/Main_Page). For the main experiment, the English test splits were used. The STS-B dev split was additionally used for some other experiments.

The SICK-R dataset is available [here](https://marcobaroni.org/composes/sick.html).

For GLUE and SuperGLUE tasks, we download and use the data programatically via the ``datasets`` library. Alternatively, they are also available for download on the official websites for [GLUE](https://gluebenchmark.com/) and [SuperGLUE](https://super.gluebenchmark.com/). 

The SemCor data for the verb tense experiment is also downloaded programatically via ``nltk``, but is also available [here](http://web.eecs.umich.edu/~mihalcea/downloads.html#semcor).

The dataset used to produce results in ``src/Punctuation_experiment.ipynb`` is available [here](https://nlp.biu.ac.il/~ravfogs/resources/syntax_distillation/).

## Repository organization

All experiments present in the paper are re-implemented in ipython notebooks under ``src``. 

* ``src/functions.py`` implements the global and cluster-based approach as well as all helper functions.

* ``src/Main_experiment.ipynb`` contains the code for evaluating the cluster-based method on STS datasets via Spearman coefficient score and isotropy.

The following notebooks contain code for evaluating performance of a MLP trained on embeddings before and after isotropy enhancement, on corresponding GLUE and SuperGLUE classification tasks:
* ``src/RTE.ipynb``
* ``src/WiC.ipynb``
* ``src/MRPC.ipynb``
* ``src/BoolQ.ipynb``
* ``src/SST2.ipynb``
* ``src/CoLA.ipynb``

Finally, some more side experiments and analysis:

* ``src/Punctuation_experiment.ipynb`` analyzes the claim that removing dominant directions from punctuations and stop words helps in removing syntactical information of CWRs.

* ``src/Verb_tense.ipynb`` analyzes the claim that the cluster-based approach brings together CWRs of verbs with same meaning, even when used in different tense.

* ``src/Layer_isotropy.ipynb`` evaluates the isotropy of embeddings from different layers of transformers on STS-B dev split.

* ``src/Local_assessment.ipynb`` evaluates the isotropy of embeddings from transformers after clustering (with various number of clusters) and zero-centering on STS-B dev split.
