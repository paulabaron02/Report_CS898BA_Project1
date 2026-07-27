Project

CS 898BA – Image Analysis and Computer Vision 

Maria Paula Baron Rodriguez

**Wsu ID:** J858Q278

**Main Objective**

To develop and optimize a computer vision system capable of accurately recognizing and classifying traffic signs by combining adaptive image preprocessing techniques with a Convolutional Neural Network (CNN).

**Specific Objectives**
1. Prepare and preprocess traffic sign images from the GTSRB dataset by applying adaptive image enhancement techniques, including image resizing, histogram equalization, contrast enhancement, Gaussian filtering, and   normalization.
2. Design, train, and optimize a CNN for traffic sign recognition by evaluating different hyperparameter combinations, including learning rate, batch size, and dropout rate.
3. Evaluate and compare the performance of the baseline and optimized CNN models using computer vision metrics such as accuracy, precision, recall, F1-score, and confusion matrices, identifying the preprocessing pipeline that provides the best recognition performance.

**Introdruction** 

Traffic Sign Recognition (TSR) has become a critical component of Advanced Driver Assistance Systems (ADAS) and autonomous vehicles because it enables vehicles to detect and interpret traffic signs in real time, improving driving safety and decision-making. According to Fu and Huang (2010), TSR systems have been an active research area for decades, and a complete recognition system generally consists of three stages: traffic sign detection, tracking, and classification.

As autonomous driving technology has evolved, researchers have identified several challenges that affect traffic sign recognition, including illumination changes, weather conditions, motion blur, partial occlusions, and variations in sign appearance. According to Lim et al. (2023), these environmental factors continue to limit the performance of recognition systems and remain one of the main research challenges in the field.

Early traffic sign recognition methods relied on traditional computer vision techniques, such as color segmentation, shape analysis, handcrafted feature extraction, and machine learning classifiers including Support Vector Machines (SVMs). However, recent studies indicate that Convolutional Neural Networks (CNNs) have become the preferred approach because they automatically learn relevant visual features from image data and consistently achieve higher recognition accuracy than traditional methods.

For experimental evaluation, this project will use the German Traffic Sign Recognition Benchmark (GTSRB). According to Stallkamp et al. (2011), GTSRB is one of the most widely used benchmark datasets for traffic sign recognition research, containing more than 50,000 labeled images across 43 traffic sign classes under diverse real-world driving conditions. Based on these studies, this project will investigate how image preprocessing techniques including image resizing, color space conversion, histogram equalization, and normalization can improve traffic sign recognition performance before evaluating and comparing traditional machine learning methods with CNN-based models. This implementation is also supported by the work of Cao et al. (2019), who demonstrated that combining effective image preprocessing with CNN architectures can significantly improve both recognition accuracy and computational performance.

**Dataset preparation**

To make the Traffic Sign Recognition project, the script begins by setting a fixed random seed for Python, NumPy, and TensorFlow. This ensures that the same images are selected and the dataset is split in the same way every time the code is executed, making it easier to compare different experiments and model configurations.

The dataset is organized into folders, where each folder represents a traffic sign class. The script scans each folder, reads all supported image formats, and verifies that every image can be loaded correctly with OpenCV. Any unreadable or corrupted files are skipped to avoid errors during training. Information about every valid image is then stored in a Pandas DataFrame called Dataset_DataFrame.

To prepare the data for training, the dataset is divided using stratified sampling so that each subset keeps the same class distribution. First, 20% of the images are reserved for testing. From the remaining data, another split is performed to create a validation set, resulting in approximately 60% training data, 20% validation data, and 20% testing data. Finally, the indices of each DataFrame are reset, producing three clean and independent datasets that are ready for image preprocessing and CNN training.

**Adaptive Preprocessing**

This stage prepares the images before they are used by the CNN. The pipeline can be executed in two modes: **with adaptive preprocessing** or **without adaptive preprocessing**, allowing both approaches to be compared under the same training conditions.

When adaptive preprocessing is enabled, each image is first resized to the input dimensions required by the model. The script then analyzes the image brightness and contrast. If an image is too dark, histogram equalization is applied to the V channel of the HSV color space to improve its brightness. If the image has low contrast, CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the L channel of the LAB color space to enhance local contrast while preserving image details.

After these optional enhancements, a Gaussian blur is applied to reduce image noise, and the image is converted to RGB format before being normalized to values between 0 and 1.

To ensure a fair comparison, adaptive preprocessing is applied only to the testing images, while the training and validation datasets use the same standard preprocessing pipeline. This makes it possible to evaluate whether adaptive preprocessing improves the model's performance without changing the data used for training.

Finally, the processed images are loaded into TensorFlow using the tf.data pipeline, where they are batched, shuffled when required, and prefetched to improve training efficiency.

**CNN Classification**

A baseline Convolutional Neural Network (CNN) was developed to classify the traffic sign images. The model consists of three convolutional layers with 32, 64, and 128 filters, each followed by a max-pooling layer to progressively extract image features while reducing the spatial dimensions. The extracted features are then flattened and passed through a fully connected layer with 128 neurons and a dropout layer (0.5) to reduce overfitting. The final output layer uses the Softmax activation function to predict the probability of each traffic sign class.

The model is trained using the Adam optimizer with the sparse categorical cross-entropy loss function, while accuracy is used as the main performance metric.

To improve the training process, Early Stopping monitors the validation loss and automatically stops training if the model no longer improves, restoring the best-performing weights. At the same time, Model Checkpoint saves the best version of the model during training.

After training, the final model is evaluated on the independent test dataset, reporting both the test loss and test accuracy. The trained model is also saved in .keras format for future evaluation or comparison with other models.

**Hyperparameter Optimization**

To improve the performance of the CNN, different combinations of hyperparameters were tested. The parameters included three learning rates, two batch sizes, and two dropout rates, resulting in a total of 12 experiments, For every trial the same CNN structure was used, but the learning rate, batch size, and dropout rate were changed. The training and validation datasets were also recreated using the batch size assigned to each experiment.

Each model was trained using the Adam optimizer and early stopping. Early stopping ended the training when the validation loss stopped improving, which helped reduce unnecessary training and overfitting, After every experiment the script recorded the best validation accuracy and the epoch where that accuracy was reached. The results were stored in a Pandas DataFrame and sorted from the highest to the lowest validation accuracy.

The model with the best validation accuracy was saved as Best_Optimized_CNN.keras. The complete optimization results were also saved as both a CSV file and an image table, making it easier to compare the different hyperparameter combinations.

**Evaluate Optimized CNN**

After the hyperparameter search, the best saved model was loaded and evaluated using the test dataset. The script calculated the test loss and test accuracy and then generated predictions for every test image.

A classification report was created to show the precision, recall, and F1-score for each traffic sign class. A confusion matrix was also generated to show which classes were predicted correctly and where the model made mistakes. Finally, the baseline CNN and optimized CNN were compared using their test loss and test accuracy.

**Without Adaptive Preprocessing**

Without adaptive preprocessing, both models achieved a test accuracy of 99.19%. The baseline CNN had a lower test loss of 0.0142, while the optimized CNN had a test loss of 0.0298.

The optimized CNN correctly classified 246 of the 248 test images. It made only two mistakes:

One pedestrian sign was classified as Wild Animals Crossing.
One Slippery Road sign was classified as Pedestrians.

The Speed Limit, Stop, and Wild Animals Crossing classes were classified correctly in every test image. These results show that the original test images were already suitable for the CNN and that the model was able to separate the five classes very well.

**With Adaptive Preprocessing**

When adaptive preprocessing was applied to the test images, the baseline CNN achieved 95.97% accuracy, while the optimized CNN achieved 91.13% accuracy.

The largest decrease occurred in the Wild Animals Crossing class. Only 35 of the 50 images were classified correctly. Seven were classified as Pedestrians, seven as Slippery Road, and one as Speed Limit. The Speed Limit class also had six images incorrectly classified as Pedestrians.

Although adaptive preprocessing was intended to improve dark or low-contrast images, it reduced the overall performance in this experiment. This may have happened because the CNN was trained using images without adaptive enhancement but was tested using modified images. The changes in brightness and contrast may have changed some of the visual features that the model learned during training.

**Results**

| With Adaptive Preprocessing | Without Adaptive Preprocessing |
|---|---|
|<img width="3343" height="1112" alt="Classification_Report" src="https://github.com/user-attachments/assets/4886228d-5573-4b20-9eea-13d221a18846" />|<img width="3343" height="1112" alt="Classification_Report" src="https://github.com/user-attachments/assets/ba8ecb6b-416a-460c-a1e7-b4bdb9fbbe8a" />|
|<img width="2702" height="2368" alt="Confusion_Matrix" src="https://github.com/user-attachments/assets/3a274a6e-8721-49b0-b26b-996814b0486e" />|<img width="2702" height="2368" alt="Confusion_Matrix" src="https://github.com/user-attachments/assets/adc5f7fd-1ad7-44f1-bf98-2923fd4ce71a" />|
|<img width="2850" height="1805" alt="Hyperparameter_Optimization_Results" src="https://github.com/user-attachments/assets/d7be0cff-8c79-49b1-bca6-e58130ae68fa" />|<img width="2850" height="1805" alt="Hyperparameter_Optimization_Results" src="https://github.com/user-attachments/assets/0cc54d4f-7a70-4043-bf12-ce2ce64e9a33" />|
|<img width="2106" height="881" alt="Model_Comparison" src="https://github.com/user-attachments/assets/2af29345-98d6-4f68-b291-b8af7b988671" />|<img width="2106" height="881" alt="Model_Comparison" src="https://github.com/user-attachments/assets/1a451560-8ae8-4762-acbc-e6a8d3f2197b" />|


**Conclutions**

The best overall results were obtained without adaptive preprocessing. The model reached approximately 99.2% test accuracy, compared with 91.1% when adaptive preprocessing was enabled.

The hyperparameter optimization produced very high validation accuracy in both experiments. However, the optimized model did not always perform better than the baseline model on the test data. This shows that the model with the highest validation accuracy is not automatically the model with the best test performance.

Based on these results, the standard preprocessing pipeline was more effective for this dataset. A future experiment could apply adaptive preprocessing to the training, validation, and testing images instead of applying it only to the test set.

**References**

Cao, Y., Wang, X., Yang, Z., Ye, X., & Wang, X. (2019). An improved convolutional neural network for traffic sign recognition. Procedia Computer Science, 147, 166–171. https://doi.org/10.1016/j.procs.2019.01.210

Fu, H., & Huang, K. (2010). A survey of traffic sign recognition. In 2010 International Conference on Wavelet Analysis and Pattern Recognition (ICWAPR) (pp. 119–124). IEEE. https://doi.org/10.1109/ICWAPR.2010.5576320

Lim, K., Kim, H., Lee, J., & Kim, C. (2023). Deep learning-based traffic sign recognition: A survey. Sensors, 23(3), 1457. https://doi.org/10.3390/s23031457

Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2011). The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In The 2011 International Joint Conference on Neural Networks (IJCNN) (pp. 1453–1460). IEEE. https://doi.org/10.1109/IJCNN.2011.6033395
