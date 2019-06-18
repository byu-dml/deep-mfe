# deep-mfe

## Steps:

### Metafeature Comparison Task
- [ ] Decide which sets of metafeatures to use
  - original, model-based (knn, perceptron, etc), graph-based, time-based, etc
- [ ] Decide which datasets to use
  - UCI, synthetic, D3M, etc
- [ ] Compute all metafeatures for all datasets
- [ ] Decide pipeline style (single classifier? fixed structure? dynamic pipelines?)
- [ ] Decide on meta-task
  - algorithm selection? pairwise comparison? hyper-parameter optimization?
- [ ] Run meta-task with various subsets and feature selected metafeatures

### Deep-learned Metafeatures task
- [ ] Decide on deep-mfe architecture
  - double attention, generative, invertible network generator, double PCA/LDA
- [ ] Attach to previous meta-task
- [ ] Run meta-task with deep-learned metafeatures
- [ ] Compare results
  - hand-crafted mfs, dataset2vec, 
