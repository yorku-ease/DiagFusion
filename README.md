# Getting Started

## Environment
Python 3.7.13, PyTorch 1.10.0, scikit-learn 1.0.2, fastText 0.9.2, and DGL 0.9.2 are suggested.

## Dataset
D1: https://github.com/CloudWise-OpenSource/GAIA-DataSet

D1 contains two datasets: MicroSS and Companion Data. We use MicroSS, for it provides trace, log, and metric at the same time.

## Demo
We provide a demo. Please run:
```
python main.py --config gaia_config.yaml
```

## Parameter Description in the Demo
### fastText \& Instance Embedding
* `vector_dim`: The dimension of event embedding vectors. (default: 100)
* `sample_count`: The number of samples per type after data augmentation. (default: 1000)
* `edit_count`: The number of events modified per sample during data augmentation. (default: 1)
* `minCount`: The minimum number of occurrences of the event (events that occur less than this number are ignored). (default: 1)
### DGL
* `epoch`: Training rounds. (default: 6000)
* `batch_size`: The number of samples contained in a batch of data. (default: 1000)
* `win_size`: The length of the judgment window for ending training early. (default: 10)
* `win_threshole`: The thresh for ending training early. (default: 0.0001)
* `lr`: The learning rate. (default: 0.001)

# Result
Based on the anomalies provided in the repo, with provided trace, logs and produced metrics using metric_event.py while taking metric points from 40 minutes before and 10 minute after
```
Top1-5:  [0.25688073 0.40672783 0.5412844  0.63302752 0.6941896 ]
anomaly type
Weighted precision 0.7359995572841643
Weighted recall 0.6605504587155964
Weighted f1-score 0.6699504724467388
test ends at 1732758634.580112
test use time 0.2241530418395996 s
```

Based on our dataset
Tracking metrics 10 min before and after
Tracking traces 31 min before and 1 min after
```
Top1-5:  [0.14136126 0.2460733  0.34031414 0.41884817 0.52879581]
anomaly type
Weighted precision 0.9073589296102386
Weighted recall 0.8638743455497382
Weighted f1-score 0.8825049337625454
test ends at 1732987689.4046998
test use time 0.15334177017211914 s
```

Based on our dataset
Tracking metrics 10 min before and after
Tracking traces 1 min before and 1 min after
```
Top1-5:  [0.08900524 0.18848168 0.34031414 0.44502618 0.53403141]
anomaly type
Weighted precision 0.9334746114850826
Weighted recall 0.8795811518324608
Weighted f1-score 0.9029214496030679
test ends at 1732987925.178825
test use time 0.1287250518798828 s
```

Based on our dataset
Tracking metrics 11 min before and after
Tracking traces 1 min before and after
```
Top1-5:  [0.08900524 0.18848168 0.34031414 0.44502618 0.53403141]
anomaly type
Weighted precision 0.9334746114850826
Weighted recall 0.8795811518324608
Weighted f1-score 0.9029214496030679
test ends at 1732987925.178825
test use time 0.1287250518798828 s
```

Based on our dataset
Tracking metrics 5 min before and after
Tracking traces 31 min before and 1 min after
```
Top1-5:  [0.15183246 0.23560209 0.32984293 0.42931937 0.51308901]
anomaly type
Weighted precision 0.8810874262444942
Weighted recall 0.8795811518324608
Weighted f1-score 0.878878599093927
test ends at 1733066916.113528
test use time 0.19760990142822266 s
```