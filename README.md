# Tri-CRLAD
This repository supplements our paper:"Semi-supervised Anomaly Detection via Adaptive Reinforcement Learning-Enabled Method with Causal Inference"

## Installation
pip install -r requirements.txt

## Datasets
| Name          | D   | Class         | Size            | Data Sources                                                |
|---------------|-----|---------------|-----------------|-------------------------------------------------------------|
| annthyroid    | 6   | abnormal      | 534 (7.4%)      | [http://odds.cs.stonybrook.edu/annthyroid-dataset/](http://odds.cs.stonybrook.edu/annthyroid-dataset/)     |
| cardio        | 21  | abnormal      | 176 (9.6%)      | [http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/](http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/) |
| satellite     | 36  | abnormal      | 2036 (31.6%)    | [http://odds.cs.stonybrook.edu/satellite-dataset/](http://odds.cs.stonybrook.edu/satellite-dataset/)       |
| satimage2     | 36  | abnormal      | 71 (1.2%)       | [http://odds.cs.stonybrook.edu/satimage-2-dataset/](http://odds.cs.stonybrook.edu/satimage-2-dataset/)     |
| Multi_cardio  | 21  | suspect pathologic | 295 (13.9%) \| 176 (8.28%) | [https://archive.ics.uci.edu/ml/datasets/Cardiotocography](https://archive.ics.uci.edu/ml/datasets/Cardiotocography) |
| Multi_har     | 561 | upstairs downstairs | 1544 (15%) \| 1406 (13.7%) | [https://www.openml.org/d/1478](https://www.openml.org/d/1478) |
| Multi_annthyroid | 21 | hypothyroid subnormal | 93 (2.5%) \| 191 (5.1%) | [https://www.openml.org/d/40497](https://www.openml.org/d/40497) |

## Run
```bash
python Cmain.py --TC 6 --TH 0.8 --task_name [DATA_NAME]
```
## Contact
aoudsung@gmail.com
