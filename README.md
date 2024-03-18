# Product-Classification-and-Clustering
Authors: [Wiktor Wierzchowski](https://github.com/wierzchw), [Wojciech Grabias](https://github.com/WojtekGrbs) <br>
Unsupervised Learning project trying to aggregate the data into clusters of products of the same category (Mobile Phones, Freezers etc.) based on their name, registered in the database as strings.
##### [Data source](https://www.kaggle.com/datasets/lakritidis/product-classification-and-categorization)

## Preprocessing
Included in `Preprocessing.ipynb`, includes deleting the pre-made labels of products, homogenizing product names in terms of structure. It also includes `jacc_sim` and `lex_sim` functions that are the core of creating the model. The file also presents the process of data exploration and eventually wraps the entire preprocessing into `preprocess()` function that allows reconstruction of the entire procedure.
### Approach
The number of clusters is determined independently of the pre-made labels - based off the silhuette charts involving `sklearn`'s `K-Medoids++` model. The metric used in both number of clusters and the eventual results was created by the authors - it is a linear combination of the Jaccard Similarity:
$$\text{Jaccard Similarity} (A, B) = \frac{|A \cap B|}{|A \cup B|}$$
and Damerau-Levenshtein Similarity:
```math
\text{DL}(s, t) = \begin{cases} 
\max(|s|, |t|) & \text{if } \min(|s|, |t|) = 0 \\
\min \left\{
\begin{array}{l}
\text{DL}(s[2:], t[2:]) + \text{substitution cost}(s[1], t[1]), \\
\text{DL}(s[2:], t) + 1, \\
\text{DL}(s, t[2:]) + 1
\end{array}
\right\} & \text{otherwise}
\end{cases}
```
## Model and results
Both the implementation of our model and its results are presented in `Model.ipynb` file. 
Final version of the model generated 10 clusters, the results of the division and its credibility is checked by creating a map of words for each of the cluster, presenting the most frequently occuring words as the biggest on the map. Examples of the clusters are as follows:
##### Mobile Phones
![42da15d4d24125984bfb8024337f06a9](https://github.com/WojtekGrbs/Product-Classification-and-Clustering/assets/51636941/d7073430-7f49-4099-96cc-eb38b89416b0)
##### Processors
![d7b530fca81886f583d40914338ba4af](https://github.com/WojtekGrbs/Product-Classification-and-Clustering/assets/51636941/057777f4-18d2-4567-a60a-4a00f928b126)
##### TVs
![3d558a0057dd029d87187b1aaf2fc314](https://github.com/WojtekGrbs/Product-Classification-and-Clustering/assets/51636941/1a8bfd41-9647-4e01-b051-e9923d7ce95b)
##### Cameras
![b84d88af158dce56dd57cba7e5639e9b](https://github.com/WojtekGrbs/Product-Classification-and-Clustering/assets/51636941/f1cab314-6e9b-43d8-a545-e09fe68922ad)

