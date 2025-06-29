# Machine Learning Tutorials
This repository offers a topic-wise curated collection of tutorials, articles, and resources on machine learning.

## Contents
- Introduction
- Training and prediction
- Traditional and Representation Machine Learning Models
- Supervised and Unsupervised Learning
- Machine Learning Workflow
- Machine Learning Library
- Scikit-Learn
- Choosing the Right Estimator


## Introduction
At its core, a machine learning model is built using an algorithm that learns from data. This algorithm can parse an input dataset, find patterns, and use these patterns to make predictions. Machine learning algorithms are designed to handle vast amounts of data, often involving millions of records, and can make intelligent decisions based on the patterns they identify.

For example, a machine learning algorithm can analyze emails on a server to determine whether they are spam or legitimate (ham). Another example is image recognition, where the algorithm identifies edges, colors, and shapes in images to determine if the image is of a girl, bird, or cat.

Machine learning is a constantly evolving field, with new types of algorithms being developed regularly, such as reinforcement learning and generative adversarial networks.

Four most common categories of machine learning problems are as follows:

1. __Classification__: This involves determining the class or category of an input. Examples include identifying whether an email is spam or ham, whether a review is positive or negative, or whether an image is of a cat, dog, or bird.
2. __Regression__: This is used to predict a continuous value. For instance, predicting the price of an automobile based on its make, model, engine size, and top speed
3. __Clustering__: This involves finding logical groupings in a large corpus of data. For example, categorizing newspaper articles into sports, entertainment, and current affairs. The key difference between classification and clustering lies in whether or not the data has labels
4. __Dimensionality Reduction__: This technique reduces the number of attributes or features in the data to find significant factors. For example, identifying the most significant factors that drive stock prices from a vast array of inputs.

## Training and prediction

Building a machine learning model involves two phases: __training__ and __prediction__. 

During the training phase, the algorithm is fed a large corpus of labeled data (e.g., reviews marked as positive or negative). The algorithm adjusts its model parameters based on the data to improve predictions. 

Once trained, the model can be used for prediction, classifying new instances it hasn't seen before.

Machine learning models learn from data to make predictions, and their accuracy depends on the quality of the input features. This approach allows models to adapt to new data without constant re-coding

Inputs to the model, called feature vectors or x variables, represent the attributes of the instances. The model's output, or prediction, is referred to as a label, y value, or predicted value

## Traditional and Representation Machine Learning Models:

__Traditional Machine Learning Models__ that rely on manually selected features from data to make predictions.

Example: You give a model inputs like age, salary, and years of experience to predict if someone will leave a company.

Traditional machine learning models rely on a fundamental algorithmic structure to solve problems. For example, a decision tree builds a tree-like structure to classify instances, while regression models fit a line or curve to make predictions. Naive Bayes models apply probabilities to input data and output probabilities. These models depend on experts to determine the features and structure of the data.

__Representation Machine Learning Models__ Models that automatically learn useful features from raw data during training.

Example: A company receives thousands of support tickets every day in free-text form,The goal is to automatically classify each ticket into categories.

As data grows larger and use cases become more complex, representation-based machine learning models, such as deep learning models, become more relevant. These models do not require experts to specify features; instead, they learn significant features from the data itself. Deep learning models, particularly neural networks, are the most common examples of representation-based machine learning.

Comparing traditional and representation-based machine learning:
- Traditional ML models rely on experts to choose features, while deep learning models learn features implicitly.
- Traditional ML models work with structured data, whereas deep learning models can handle unstructured data like images and videos.
- Both types of models can solve common machine learning problems, but traditional ML models offer more insight into their mechanics, while deep learning models are often black-box models.

## Supervised and Unsupervised Learning

In machine learning, there are two primary types of algorithms: supervised and unsupervised learning. 

__Supervised learning__ involves training data with labels, which are used to correct the algorithm. This approach is common for beginners and includes techniques like classification and regression. Linear regression is a simple example of supervised learning

__Unsupervised learning__, on the other hand, does not have labeled training data. The model must learn the structure in the data on its own. This approach is used to find patterns and groupings in the data. Common techniques include clustering (e.g., K-means clustering) and dimensionality reduction (e.g., Principal Component Analysis or PCA). Unsupervised learning is often used to preprocess data for supervised learning tasks.

## Machine Learning Workflow:

1. Understand the problem:
    - Define the task clearly (e.g., spam detection, image classification, stock prediction).
2. Gather relevant data:
    - Collect raw data from sources like the web, APIs, or databases.
    - Data may come in different formats.
3. Prepare and preprocess the data:
    - Load data into a usable format (e.g., CSVs, cloud storage).
    - Clean and format the data.
    - Convert non-numeric values (e.g., strings or categories) into numeric format.
4. Select and train a model:
    - Choose a suitable machine learning algorithm (e.g., decision trees, SVMs).
    - Train the model using the prepared data.
5. Validate the model:
    - Evaluate the model’s performance using validation techniques.
    - Assess accuracy and real-world applicability.
    - If results are unsatisfactory, iterate by adjusting the algorithm, data, or training.
6. Deploy the model:
    - Launch the trained model into production.

![Basic Machine Learning Workflow](images/machine-learning-workflow.png)

## Machine Learning Library

- Machine learning libraries provide pre-built functions that simplify complex tasks, speeding up the development process.
- They allow developers to design and deploy models quickly, without needing to code from scratch.
- These libraries offer scalability, enabling models to handle large datasets and complex computations efficiently.
- By using these libraries, developers can focus on high-level tasks instead of reinventing foundational algorithms.
- The extensive use of ML libraries enhances the flexibility and adaptability of AI applications, driving faster innovation.

When selecting a machine learning library, it’s crucial to consider the specific requirements of your use case, such as the type of model, data size, and performance needs. Choose a library that aligns with your goals, whether it’s for quick prototyping, scalability, or specialized algorithms.

Here are some popular machine learning libraries:

- Scikit-Learn
- TensorFlow
- PyTorch
- Keras

## Scikit-Learn

Scikit-learn provides APIs for every step of the machine learning process, all built using a higher-level estimator object. This consistent interface is used not only for machine learning algorithms but also for preprocessing techniques, model selection, and evaluation methods. 

The basic steps involve creating a model object, fitting it to the training data using the fit or fit_transform method, and then using the trained model for predictions on new data. The estimator API is also integral to constructing complex pipelines.

The design principles of the estimator API include consistency, straightforward inspection of model parameters, limited object hierarchy, and sensible defaults for input parameters. 

Scikit-learn interoperates seamlessly with Pandas and NumPy, allowing you to load raw data in these formats directly into scikit-learn.

The sklearn.preprocessing namespace offers functions for standardization, normalization, scaling, and converting categorical data to numeric form. It also provides techniques to handle missing values, outliers, and noisy data.

Scikit-learn has a comprehensive suite of algorithms for regression, classification, clustering, and dimensionality reduction, all exposing a consistent interface for training models.

Validation is crucial to ensure model performance, and scikit-learn provides cross-validation tools and metrics for different models, such as R squared for regression and accuracy for classification. 

Pipelines in scikit-learn allow you to train and evaluate models efficiently, caching intermediate results and enabling fine-grained control over the workflow. Pipelines can include cross-validation techniques and be tuned as a whole.

Reference: [Scikit-Learn](https://scikit-learn.org/stable/)

## Choosing the Right Estimator

When exploring scikit-learn, a variety of machine learning techniques and estimator objects are available. Selecting the appropriate estimator begins with clearly defining the problem to be solved, which is essential before proceeding to implementation. The scikit-learn documentation includes a helpful flowchart to assist in choosing the right estimator.

The scikit-learn flowchart can then be used to guide the selection of an appropriate estimator based on the specific task. This approach supports effective navigation of the available machine learning techniques and informed decision-making.

![Algorithm cheat sheet](images/ml-map.svg)

__Steps for Choosing a Classification Algorithm with scikit-learn__
1. Check data availability
   - Ensure there are at least 50 correctly labeled training samples.
   - If not, gather more data, as machine learning is ineffective with very small datasets.
2. Define the problem type
   - Determine the goal of the task.
   - If the task involves predicting categories (e.g., classifying reviews as positive or negative), it is a classification problem.
3. Select an algorithm based on dataset size
   - For medium-sized datasets (fewer than 100,000 samples):
     - Start with a linear __Support Vector Classifier (SVC)__.
   - If the linear SVC is ineffective:
     - For text data, try a __Naïve Bayes classifier__.
     - For non-text data, consider using the __KNeighbors__ classifier.
   - If these do not yield satisfactory results:
     - Explore __general-purpose SVCs__ or __ensemble classifiers__ (e.g., Random Forest, Gradient Boosting).
4. For large datasets (more than 100,000 samples):
   - Begin with the __Stochastic Gradient Descent (SGD)__ classifier.
   - If needed, apply kernel approximation techniques to transform data for better performance with linear models.

__Steps for Choosing a Clustering Algorithm with scikit-learn__
1. Check for labeled data
   - If correctly labeled training data is not available and the goal is to predict a category or label, use clustering techniques to find patterns in the data
2. Determine if the number of categories is known
   - Known categories (e.g., 3 categories for cat, dog, bird):
     - If the dataset has fewer than 10,000 samples:
       - Start with __K-Means clustering__.
       - If K-Means is not effective, try __Spectral Clustering__ or __Gaussian Mixture Models__.
     - If the dataset has more than 10,000 samples:
       - Use __MiniBatch K-Means__, which is designed for scalability with large datasets.
   - Unknown Categories
     - If the dataset has fewer than 10,000 samples:
       - Use __MeanShift__ or __VBGMM (Variational Bayesian Gaussian Mixture Models)__ to automatically identify clusters.
     - If the dataset has more than 10,000 samples:
       - scikit-learn __may not provide suitable__ clustering options for this case.

__Steps for Choosing a Regression Model with scikit-learn__

1. Determine dataset size and dimensionality
   - If the dataset has fewer than 100,000 samples and low dimensionality (only a few important features), consider using:
     - __Lasso regression__ or __ElasticNet regression__.
2. Handle high-dimensional data
   - For datasets with many significant features (high dimensionality), try:
     - __Ridge regression__ or __Support Vector Regression (SVR)__ with a linear kernel.
3. Explore alternative models if initial results are unsatisfactory
   - If the previous models don’t perform well, consider:
     - __SVR with a different kernel__ (e.g., RBF kernel).
     - __Ensemble regressors__ (e.g., Random Forest Regressor, Gradient Boosting).
4. For very large datasets
   - If the dataset has more than 100,000 samples, use:
     - __Stochastic Gradient Descent (SGD) regressor__, as it is suitable for large-scale data.

__Steps for Using Dimensionality Reduction with scikit-learn__
1. Identify the goal
   - If the goal is not predicting a quantity, but rather exploring the data to identify significant features or latent factors, use dimensionality reduction techniques.
2. Start with Principal Components Analysis (PCA)
   - Begin with __Principal Components Analysis (PCA)__, which is widely used for reducing data dimensions while retaining important features.
3. Explore alternatives if PCA is ineffective
   - If PCA doesn’t produce good results, consider these options for datasets with fewer than 10,000 records:
     - __Isomap__.
     - __Spectral Embedding__.
4. Handle large datasets
   - For larger datasets, use __kernel approximation techniques__ to efficiently reduce dimensions.

__Note__: If the task does not involve predicting a category, predicting a quantity, or exploring data through dimensionality reduction, but instead focuses on predicting the structure of the data,   scikit-learn does not provide specific libraries for this purpose

