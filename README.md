# DSS5104-MolecularPropertiesGNN

## Overview
Drug discovery plays a critical role in the development of new therapies and in ensuring the safety and efficacy of chemical compounds. A key step in this process is the prediction of molecular properties, such as biological activity and toxicity, to quickly and accurately identify appropriate drug candidates. Traditionally, computational approaches in cheminformatics have relied heavily on hand- crafted descriptors, such as molecular fingerprints, to represent chemical compounds. Although these methods have proven effective in various applications, their ability to capture the intricate relation- ships within molecular structures is inherently limited. Recent advances in deep learning, in particular, graph neural networks (GNNs), provide another idea that models learn directly from molecular graphs. <br>
In this project, we systematically implements both classical machine learning and GNN approaches using the Tox21 dataset. The dataset comprises approximately 8,000 compounds, containing a SMILE column (Simplified Molecular-Input Line-Entry System string, a standardized textual format for encoding molecular structures) and 12 columns of binary toxicity endpoints, corre- sponding to various biological targets associated with toxicological responses, used for model fitting. We used the Area Under the ROC Curve (AUC ROC) for each of the 12 targets as the main evaluation metric, combined with some classification indices (e.g., F1 score, recall indicator) for additional supple- mentation. In doing so, our objective was to assess the strengths and limitations of these methodologies in predicting molecular activity.


## Group Member
|      Name      |       Contact       |
| -------------- | ------------------- |
| Ding Jiaming   | e1351662@u.nus.edu  |
| Niu Muyuan     | e1352057@u.nus.edu  |
| Li Jingming    | e1352254@u.nus.edu  |
| Zhang Yi       | e1351350@u.nus.edu  |


## Candidate Models
### 1. Classic Machine Learning Methods
* XGBoost
* LightGBM
### 2. Graph Neural Network
* GCN
* D-MPNN
* RNN <br>
<b> In this project, we also explored enhanced versions of the base model to further improve its discriminative capability beyond just AUC performance.<b>
<br>

## Comaprisons
| Model                      | Average AUC |
|---------------------------|-------------|
| RNN-BiLSTM                | 0.84        |
| D-MPNN                    | 0.83        |
| D-MPNN with Focal Loss    | 0.83        |
| GCN                       | 0.82        |
| GCN with Lightning        | 0.82        |
| XGBoost                   | 0.80        |
| LightGBM                  | 0.78        |


