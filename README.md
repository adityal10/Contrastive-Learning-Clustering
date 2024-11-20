# Contrastive Learning with Image Clustering
This project implements a Contrastive Learning architecture using SimCLR in PyTorch. The model learns meaningful embeddings by contrasting positive and negative pairs of images. The learned embeddings are then clustered using K-Means to evaluate their performance in unsupervised image classification.

## Features
* Implements SimCLR contrastive learning from scratch.
* Uses ResNet-18 as the backbone for feature extraction.
* Evaluates embeddings using K-Means clustering and computes metrics such as:
    *  Normalized Mutual Information (NMI)
    * Adjusted Rand Index (ARI)
* Visualizes clustering performance using t-SNE for dimensionality reduction.

## Dataset
* CIFAR-10 is used as the dataset, containing 60,000 images in 10 categories.
* Images are preprocessed with data augmentations during training and normalized during evaluation.

## Results
* The model achieves clear separation of clusters, as shown in the t-SNE plot (`refer to clusters_visualization.png`).
* Metrics:
    * NMI: ~0.7 (depends on training setup)
    * ARI: ~0.6 (depends on training setup)


## How to Run the Code

1. Clone the repository:

```bash
git clone https://github.com/your-username/contrastive-learning-clustering.git
cd contrastive-learning-clustering
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the SimCLR model:

```bash
python train_simclr.py
```

4. Cluster and evaluate embeddings:

```bash
python contrastive_learning_with_clustering_and_visualization.py
```

5. View Clustering Results:
* Check `clusters_visualization.png` for the t-SNE visualization.

## Files in the Repository
* `train_simclr.py`: Script to train the SimCLR model.
* `contrastive_learning_with_clustering_and_visualization.py`: Evaluates embeddings using K-Means and visualizes clusters.
* `requirements.txt`: Python dependencies.
* `README.md`: Documentation for the project.


## Future Improvements
* Use advanced clustering methods (e.g., DBSCAN, hierarchical clustering).
* Test on larger datasets (e.g., ImageNet) for better generalization.
* Extend visualization by overlaying image thumbnails.

## License
This project is licensed under the MIT License.