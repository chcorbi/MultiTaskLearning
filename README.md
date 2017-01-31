#Multi-Task Learning
Authors : Charles Corbière, Hamza Cherkaoui

## Synopsis

This package implement differents multi-task learning models:
- Multilearning SVM (svm): an SVM is learning for each task
- Alternating Structure Optimization (aso): a modele assuming every task shared a low dimensional structure
- Convex Alternating Structure Optimization (caso): convex relaxation of ASO
- Clustered Multi-task Learning (cmtl): a modele assuming tasks are groupd within clusters.

Dataset included:
- a clustered toy dataset (toy)
- School data (school)
- Sarcos data (sarcos)


## How to use it

- To compute score for a given algorithm on a given dataset, for a test size proportion and a number of splits
```
python computeScores.py school cmtl 5 0.30
```
Here, we run 5 times CMTL on school dataset with a 30% test size proportion.

- To plot all algorithms scores for a given dataset and a number of splits, iterating on the test size proportion
```
python plotResults.py school 5
```
Here, we run 5 times for each algorithm on school dataset. Note that on current implementation, the test size range is [0.30, 0.40, 0.50, 0.60]




