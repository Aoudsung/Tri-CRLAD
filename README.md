# Tri-CRLAD
This repository supplements our paper:"Semi-supervised Anomaly Detection via Adaptive Reinforcement Learning-Enabled Method with Causal Inference"

## Installation
pip install -r requirements.txt

### Datasets
\begin{table}[ht]
    \centering
    \caption{Description of Datasets}
    \scalebox{0.8}{%
        \begin{tabular}{ccccl}
            \toprule
            \multicolumn{2}{c}{Dataset} & \multicolumn{2}{c}{Anomaly Size} & \multicolumn{1}{c}{\multirow{2}*{Data Sources}} \\
            \cmidrule{1-4}
            Name  &   D  &  Class  & Size  &  \\
            \midrule
            annthyroid & 6   &abnormal& 534 (7.4\%)   & \url{http://odds.cs.stonybrook.edu/annthyroid-dataset/} \\
            cardio  &   21  &abnormal& 176 (9.6\%)  & \url{http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/} \\
            satellite & 36 &abnormal& 2036 (31.6\%) & \url{http://odds.cs.stonybrook.edu/satellite-dataset/} \\
            satimage2 & 36 &abnormal& 71 (1.2\%) & \url{http://odds.cs.stonybrook.edu/satimage-2-dataset/} \\
            \midrule
            Multi\_cardio & 21 & \begin{tabular}{@{}c@{}}suspect \\ pathologic \end{tabular}&\begin{tabular}{@{}c@{}}295 (13.9\%)\\  176 (8.28\%)\end{tabular} & \url{https://archive.ics.uci.edu/ml/datasets/Cardiotocography} \\
            Multi\_har & 561 & \begin{tabular}{@{}c@{}}upstairs \\ downstairs \end{tabular} & \begin{tabular}{@{}c@{}}1544 (15\%) \\ 1406 (13.7\%)\end{tabular} & \url{https://www.openml.org/d/1478} \\
            Multi\_annthyroid & 21 & \begin{tabular}{@{}c@{}}hypothyroid\\ subnormal \end{tabular}&\begin{tabular}{@{}c@{}}93 (2.5\%) \\ 191 (5.1\%)\end{tabular} & \url{https://www.openml.org/d/40497} \\
            \bottomrule
        \end{tabular}
    }
    \label{tab1}
\end{table}
## Run
```bash
python Cmain.py --TC 6 --TH 0.8 --task_name [DATA_NAME]
```
## Contact
aoudsung@gmail.com
