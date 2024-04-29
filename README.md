"# Casting-Image-Classification-using-CNN" 

Casting Image classification
1.	Introduction:
Casting defect detection is crucial in the manufacturing industry to ensure the quality of products. Manual inspection processes are time-consuming and prone to errors. Hence, automating this process using deep learning classification models can significantly improve efficiency and accuracy. In this report, we discuss the implementation of such a model for detecting defects in casting products.

2. Dataset Description:
The dataset consists of 7348 grayscale images of submersible pump impellers, categorized into two classes: Defective and Ok. Each image is of size 300x300 pixels. The dataset is split into training and testing sets with a specified number of images in each class for both sets.

3. Data Preprocessing:

Image data is loaded using TensorFlow's image_dataset_from_directory function.
Data augmentation is applied to the training set to increase the diversity of training samples.
Images are normalized by def , ok as [0, 1].
4. Model Architecture:

The model is built using the Sequential API of Keras.
It consists of convolutional layers followed by batch normalization and max-pooling layers.
The final layers include fully connected (dense) layers with dropout regularization to prevent overfitting.
The output layer has a sigmoid activation function since this is a binary classification problem.
5. Model Training:

The model is compiled with the Adam optimizer and binary cross-entropy loss function.
Training is performed for 20 epochs.
Training and validation accuracy and loss are monitored during training.


6. Model Evaluation:

The model achieves high accuracy on the training set, with accuracy reaching approximately 97.8%.
Validation accuracy is also high, peaking at around 99.0%.
Both training and validation loss decrease consistently over epochs, indicating effective learning.
However, there are fluctuations in validation accuracy and loss, suggesting possible overfitting.
7. Insights:

The model achieves a high training accuracy of approximately 97.8%.
Validation accuracy peaks at around 99.0%, indicating that the model generalizes well to unseen data.
Fluctuations in validation metrics may indicate the need for further regularization techniques or hyperparameter tuning to improve model stability.
Overall, the model's performance suggests its potential for automating the casting defect detection process, reducing manual effort, and improving efficiency in the manufacturing industry.
8. Conclusion:
In conclusion, the developed deep learning classification model demonstrates effective detection of casting defects in submersible pump impellers. Further optimization and fine-tuning could enhance the model's robustness and reliability for real-world applications in industrial settings.

Explanation of Code Line by Line:

Import necessary libraries and modules for data processing and model building.
Load the dataset from the provided directory using image_dataset_from_directory.
Define a function for data normalization (process) and apply it to both the training and validation datasets using the map function.
Build the model architecture using the Sequential API, consisting of convolutional layers, batch normalization, max-pooling layers, and fully connected layers with dropout regularization.
Compile the model with the Adam optimizer, binary cross-entropy loss function, and accuracy metric.
Train the model using the fit function, specifying the training dataset, validation dataset, and number of epochs.
Plot the training and validation accuracy over epochs using matplotlib.
