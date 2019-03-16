# Project Specification {-}

# Formalities

## Preliminary Thesis Title

Using clustering as an unsupervised method for song categorization and music genre exploration. 

## Name and Email

Emil Juzovitski - emiljuz@kth.se

## Supervisor @CSC

Johan Gustavsson - johangu9@kth.se

## Name of the principal and name of the supervisor at the principal's workplace

* _(Principal)_ Soundtrack Your Brand

* _(Supervisor)_ Omar Marzouk - Omar@soundtrackyourbrand.com

## Current Date
Wednesday, January 23, 2019

\pagebreak
# Background & Objective

## Description of the area within which the degree is being carried out.

This thesis will revolve around the research topic of clustering analysis. Clustering analysis is the term used for unsupervised learning techniques that tries to group data of multiple dimensions together if they are similar.

One of the biggest changes in the landscape of software during the last decade is automation of features. Features that was once done manually by the end-user. Clustering is one way of allowing automation. While an end-user might not in a literal sense try to group datapoints, he might be searching for similar datapoints. By creating clusters we reduce the time needed for the end-user to find what they are looking for.

In this thesis a focus will be on clustering _songs_ on the order of the principal. The principal is a _Music Streaming Platform_ company specializing in providing music for retail businesses. The platform saves the customer time by automating the music selection. The retail customer picks characteristics e.g. Energy, Genre, Sound that fit their brand. Sets of songs (play-lists) that fit the chosen characteristics are generated for the customer to play.

The construct for the thesis is that with the tool of clustering, similar songs can be grouped together. The clusters might uncover themes in songs. _Music Experts_ that work professionally with creating play-lists for the company, will decide whether there is a theme or not in a cluster.
If a cluster is determined to have a theme, songs of that cluster can be assigned a categorical label, that is, the theme. Given enough themes, the previously unlabeled dataset can become labeled, and labeled with new, previously unknown music themes. A resulting effect is that the data can be used for supervised learning. Additionally, clustering songs can help _Music Experts_ to find candidate songs for tailored play-lists.

A dataset of _songs_ will be used. It includes more than 50 million entries with each song entry holding hundreds of attributes. Attributes of a song is its metadata as well as its audio embedding: an output from a machine learning algorithm characterizing the audio of the song. Not all attributes are interval-scaled(continuous-linear). The metadata attributes are e.g. Release year(Ordinal), BPM(Ordinal), Explicit(Binary), Artist Gender(Binary). The embedding is a vector of audio properties of the song with dimensionality in the hundreds. What each attribute means is unknown, the value of the attribute is interval-scaled. 

As stated previously, the data is mixed. The attributes are of different data types. While interval-scaled distances can be measured well through the euclidean distance measurement, other data types e.g. Ordinal data cannot be measured in the same fashion without information loss. The _Generalized Gower distance(daisy function)_ provides a way to measure mixed attributes however, many clustering packages only allow for euclidean distance. 

Given the dataset of mixed-data attributes, the thesis will try to cluster the data. A main topic of the thesis is to compare different clustering algorithms and evaluate which algorithm is most suitable for the given dataset both in terms of clustering quality(using the silhouette coefficient) and scalability(time to cluster different sample sizes of the dataset). Other topics include preprocessing and sampling from the clusters.


## Principal's interest

There are two main interests of the company:

* The company wants to find a way to efficiently create supervised training data. Clustering datapoints/songs would allow to categorize data and append a approximate label to each data point. Supervised training data would unlock the usage of supervised learning for further projects. 

* Play-lists are handpicked by _"Music Experts"_ of the company, creating play-lists requires extensive labour. Clustering can help reduce the _Searching_ done by the _experts_ by giving them _candidate songs_ for a cluster/themes, resulting in less time focused on irrelevant songs.

## Objective. What is the desired outcome (from the principal's side and from the perspective of the degree project)

> Grand scheme

From a principal standpoint the objective would be to create an algorithm which creates at least _some_ clusters that _"Music Experts"_ agree to have a theme. 

> to evaluate algorithms

From a degree project standpoint the main objective is to evaluate different clustering approaches for the given dataset. Comparing their silhouette coefficient given dataset size and time. A desired outcome is implementing a clustering approach that creates dense cluster according to the silhouette coefficient. An additional secondary outcome is that clustering and preprocessing is done in such fashion that in terms of _direct evaluation(Purity)_ our top 10-100 clusters give out a score over 50%.



# Research Question & Method

## The QUESTION that will be examined. Formulated as an explicit and evaluable question.

* **How can we use state-of-the-art clustering to categorize music, given a dataset of songs with hundreds of mixed attributes?**

* **How can we use state-of-the-art clustering to categorize music, given a dataset of songs with attributes of metadata and an audio embedding?**

* **How can we use state-of-the-art clustering to categorize music, given a dataset of songs with attributes of metadata and an audio embedding?**

* **How can state-of-the-art clustering be used to categorize music, given a dataset of songs including metadata and an audio embedding of mixed data-types?**


## Specified problem definition

In short the problem entails finding clusters through clustering analysis on a dataset on or a sample of _50 million_ songs. A property of the dataset is that the attributes are of mixed data types. The resulting clusters, should include songs of similar attribute properties, in other words: the members of a cluster should have a music theme in common. 

The challenge rises from the dataset being real-world, where a datasets are often of mixed data types as it is for this dataset. Extra steps are necessary to cluster on mixed data as the majority of clustering algorithms are implemented using the euclidean distance measurement. The steps can alternate given the approach. One challenge is to decide the approach e.g. Should Gower distance be used? Can we use a specific clustering method created for mixed data? Should we use an ensemble approach?

## Examination Method

### Method

* Preprocessing of data

    * We cannot assume all features allow clustering out of the box. Certain features need to be divided, some hashed, some just do not work(images) and so forth, some are irrelevant and need to be thrown out.

    * Some features are inadvertently more important than others, weighting is a simple solution however, it is not clear how to find what features are important without resulting to complex solutions such as using a dimension reducing neural network.

* Choosing Clustering algorithms

    * Different types of clustering algorithms will be compared _CLARANS, DBSCAN, QuickShift, ROCK_ Algorithm etc.
        * Determining what's ideal for the dataset.
        * Handling mixed data-types.
            * Distance measure approach (Gower?).
            * Finding specific mixed attribute algorithms.
            * Ensemble methods? e.g. DBSCAN with ROCK
            * Ignoring, the issue, conversion to euclidean distance.
            * Cluster on non-categorical, filter on the categorical as a post-processing step.

* Determining parameters for clustering

    * Partitioning algorithms e.g _K-means, CLARANS_ assume *k* (the amount of clusters) is known beforehand. This is not the case for this thesis and so *k* approximation is necessary (that isn't too computationally heavy). 
        * Sampling is an option. 
        * Another solution is to use another clustering approach such as density based clustering e.g. _DBSCAN_ that does not require _k_.

* Implementing Clustering algorithms

* Evaluating clustering approaches
    * Comparing the inner criterion.
    * Creating a _Purity_  test
        * Ranking clusters by silhouette coefficient
        * Sampling from top resulting clusters

### Dataset

A dataset of or a sample of 50 million songs will be used, each song including metadata and an audio embedding, in total there are hundreds of attributes. The dataset is provided by the company. A brief description of the dataset is given in the _Background_.

Google Cloud will be the computation source. To allow possible distributed implementations, code will be written in Python and, Apache Beam or Apache Spark.

## Expected scientific results
Clustering accuracy is relative. While we can measure the inner criterion e.g. silhouette coefficient a high silhouette doesn't necessary translate into clusters with themes that makes sense from an end-user standpoint. Instead direct evaluation can be used by asking the end-user about the resulting clusters and using the measurement of _Purity_.

This thesis has the resources to user _direct evaluation_ in combination with the _inner criterion_ in the context of Music and _Music Experts_. How well clustering performs on a dataset from this field is of interest of professionals and scientists within the field considering using clustering for _Music_ or other fields within _Audio_. While it is unreasonable to ask experts to evaluate the _purity_ for the whole cluster, it is reasonable to ask them to classify a sample of a cluster and determine the _purity_ on a subset of clusters.

My hypothesis is that the silhouette coefficient will always be high no matter what algorithm we use. How well they scale with data-size will differ. It is of my belief that density based clustering algorithms will perform better in terms of scalability compared to its partitioning counter-part.

# Evaluation and News Value

## Evaluation
From a degree project standpoint the objective is fulfilled if multiple candidate clustering approaches have been tested and some obtain a high silhouette coefficient with different data sample sizes.

The research question is answered by proposing a way to cluster songs of mixed attributes.

There is one main evaluation and one secondary evaluation of the method.

* The main evaluation is to look at the inner criterion: the density of the cluster e.g. silhouette coefficient of the clusters with respect to dataset sample size and time to cluster.

    * There are alternatives to silhouette coefficient.

    * The inner criterion will show how well the clustering algorithm performs.

* The secondary evaluation is determining the _Purity_ through direct evaluation done by the _Music Experts_

    * The direct evaluation tells us how well the _pipeline_ clusters the data in terms of music theme quality of the clusters.

    * Without direct evaluation it is hard to evaluate the _pipeline_ on the given dataset.


## News Value

It is often the case in real world situations that the clustering analysis is required to be done on data fetched from a database. A Database often stores data of different types and so the dataset to cluster is often of mixed data types. This thesis proposes a clustering solution for categorizing songs on a dataset of mixed attributes. Much, if not all of what is written can be extended to mixed data clustering in general.

# Pre-study

Will focus on clustering algorithms and everything around it, to be more specific: Unsupervised _Single-machine_(for now) clustering methods. 

## Finding information

### Keywords
_Clustering, Mixed data, Review, Pre-processing_

### Method

* Initially start with finding Review articles on state of the art clustering methods. 
* Go more in depth into algorithms suitable for dataset, look at original paper.
* Find sources on pre-processing.
* Search for mixed data solutions.

## Obtaining the necessary knowledge and preliminarily references

There are numerous steps in order to go from data to output:

* The first step is data type assertion and conversion. What kind of data is the dataset? How do we measure it can we convert it.

    * This topic is assessed by looking at how clustering is affected by data types and distance measurement. 

    * Current reference is _Finding Groups In Data_(Kaufmann, 1990).

* The Second step is to find suitable clustering algorithms. A way to start is to look at a review summarizing some of the higher used algorithms of today. From there the original algorithm-specific clustering papers can be viewed.

    * Current reference for summary is _Big Data Clustering: A Review_ (Shirkhorshidi, 2014).
    * Current reference for algorithm specific papers are _CLARANS: A Method for Clustering Objects for Spatial Data Mining_ (R.Ng, 2002), and, _A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise_ (Ester, 1996).


# Conditions & Schedule

## List of resources

* _"Music Experts"_ - Professional play-list makers.

    * These professionals are full-time employees at the principal. It cannot be expected that they go through all songs and all clusters. However, going through samples of 10-100 top clusters is to be expected by the Music experts.

* Google Cloud

    * Cloud computing

* Machine with over 100GB of memory

    * With Google cloud any machine can be ordered up until terabytes of memory. Clusters of machines could also be ordered.

    * For the principal it is expected that the thesis will need a machine/machines that matches the requirments from my side.

* Python Clustering Library

    * Maybe pyclustering (https://github.com/annoviko/pyclustering)
    * Maybe Spark MLLib

* Apache Beam/Apache Spark

    * A distributed framework

## Defined Limitations

I will describe the attributes of the dataset that are of interest for the thesis and its data types with the exception being the audio embedding. How the audio embedding is created will not be mentioned either due to company policy. 

I could expand my limitations with time but for now, Constraint Clustering, Neural Networks such as SOM will not be considered at the time of writing. 

## Collaboration with the principal

The principal will:

* provide the dataset.

* Provide extensive support and knowledge

    * Guide me with decisions and ML problems.

    * Discussions on results

    * Might read my report but not main objective.

    * Support on how to use the company resource eco-system

## Schedule

