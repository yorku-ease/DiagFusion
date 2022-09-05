# Getting Started

## Environment
Python 3.7.13, PyTorch 1.10.0, scikit-learn 1.0.2, fastText 0.9.2, and DGL 0.9.2 are suggested.

## Dataset
|Dataset|Data modal|
|:---:|:---:|
|GAIA|Trace|
| |Log|
| |Metric

## Demo
We provied a demo. Please run:
```
python main.py --config gaia_config.yaml
```

## Parameter Description in the Demo
### fastText \& Instance Embedding
* `vector_dim`: The dimension of event embedding vectors.
* `sample_count`: The number of samples per type after data augmentation.
* `edit_count`: The number of events modified per sample during data augmentation.
* `minCount`: The minimum number of occurrences of the event (events that occur less than this number are ignored).
### DGL
* `epoch`: Training rounds.
* `batch_size`: The number of samples contained in a batch of data.
* `win_size`: The length of the judgment window for ending training early.
* `win_threshole`: The thresh for ending training early.
* `lr`: The learning rate.

## Overview
* config: Configuration files.
    * <demo_train160.csv> Valid fault labels for GAIA dataset.
    * <gaia_config.yaml> Configuration of GAIA dataset.
* detectors: Code for anomaly detection. The main function is to complete anomaly detection on raw data.
* transforms: Code that processes data, such as log parsing. The main function is to complete data conversion or extract some kind of information.
    * events:
        * <log_parse.py>
        * <metric_ksigma.py>
        * <trace_totimeseries.py>
        * <fasttext_with_DA.py>
        * <sententce_embedding.py>
    * feature:
        * <tfidf_extractor.py>
        * <bow_extractor.py>
        * <ngram_extractor.py>
        * <tfidf_extractor.py>
* data:  Store the results of code processing in the transformers directory, not the raw data.
    * gaia: GAIA dataset.
        * events:
            * <log_*.pkl> Log template.
            * <trace_anomalies.pkl> Trace anomalies.
            * <metric_anomalies.pkl> Metric anomalies.
        * feature:
            * <tfidf_log.pkl>
            * <tfidf_metric.pkl>
            * <tfidf_log_and_metric.pkl>
            * <bow_log.pkl>
            * <bow_metric.pkl>
            * <bow_log_and_metric.pkl>
* models: Machine learning model code, such as graph neural networks, etc.
    * <He_DGL.py> Implementation of the main processes of DiagFusion.
    * <layers.py> Implementation of GNN layer.
* <public_functions.py> Public tool functions.
* <main.py> Model Entrance.
* <requirements.txt> List of dependent python libraries.

