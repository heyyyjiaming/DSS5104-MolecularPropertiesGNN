# DSS5104-MolecularPropertiesGNN

## Overview
Drug discovery plays a critical role in the development of new therapies and in ensuring the safety and efficacy of chemical compounds. A key step in this process is the prediction of molecular properties, such as biological activity and toxicity, to quickly and accurately identify appropriate drug candidates. Traditionally, computational approaches in cheminformatics have relied heavily on hand- crafted descriptors, such as molecular fingerprints, to represent chemical compounds. Although these methods have proven effective in various applications, their ability to capture the intricate relation- ships within molecular structures is inherently limited. Recent advances in deep learning, in particular, graph neural networks (GNNs), provide another idea that models learn directly from molecular graphs.
In this project, we systematically implements both classical machine learning and GNN approaches using the Tox21 dataset. The dataset comprises approximately 8,000 compounds, containing a SMILE column (Simplified Molecular-Input Line-Entry System string, a standardized textual format for encoding molecular structures) and 12 columns of binary toxicity endpoints, corre- sponding to various biological targets associated with toxicological responses, used for model fitting. We used the Area Under the ROC Curve (AUC ROC) for each of the 12 targets as the main evaluation metric, combined with some classification indices (e.g., F1 score, recall indicator) for additional supple- mentation. In doing so, our objective was to assess the strengths and limitations of these methodologies in predicting molecular activity.

## Candidate Models
### 1. Classic Machine Learning Methods
* XGBoost
* LightGBM
### 2. Graph Neural Network
* GCN
* D-MPNN
* RNN

# GCN主流程


##  配置参数
BATCH_SIZE = 32          # 每个批次的分子数量 
EPOCHS = 50              # 每个任务训练30轮
LEARNING_RATE = 0.001    # 优化器学习率
HIDDEN_DIM = 128         # GCN隐藏层维度

##  数据预处理模块
读取CSV文件

删除指定毒性指标的缺失值样本

比如当处理NR-AR任务时，只保留该列非空样本

##  分子图转换模块
分子图表示：

原子特征：6个化学相关特征

边特征：3个键特性

关键数据结构：

x: 形状为 [num_atoms, 6] 的原子特征矩阵

edge_index: 形状为 [2, num_edges] 的边连接索引

edge_attr: 形状为 [num_edges, 3] 的边特征矩阵

##  数据集类
核心作用：将原始数据转换为PyTorch Geometric需要的图数据格式

##  GCN模型类
关键操作：

图卷积：提取局部结构特征

全局池化：将原子级特征转换为分子级特征

维度压缩：将输出从 [batch_size, 1] 变为 [batch_size]

##  单任务训练函数
任务隔离：每个毒性预测作为独立二分类问题

优势：避免多任务学习的复杂参数调整

代价：需要训练12个独立模型

关键阶段：

前向传播：分子图 → GCN → 预测值

损失计算：二元交叉熵损失

反向传播：自动梯度计算

参数更新：Adam优化器调整权重


##  主执行流程
遍历所有12个毒性指标

对每个指标单独训练模型

记录每个任务的测试AUC

每个任务独立保存最佳模型 (best_model_{target}.pth)

最终输出汇总结果
