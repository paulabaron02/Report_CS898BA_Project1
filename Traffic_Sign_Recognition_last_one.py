"""
Traffic Sign Recognition
Final Project

Maria Paula Baron Rodriguez
WSU ID: J858Q278
"""

# %% IMPORT LIBRARIES

import os
import shutil
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

plt.close("all")

# %% Dataset Preparation
Random_Seed = 42

random.seed(Random_Seed)
np.random.seed(Random_Seed)
tf.random.set_seed(Random_Seed)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

Dataset_Path = r"C:\Users\Paula\.spyder-py3\MariaBaron-CS898BA-Project1\Report_CS898BA_Project1\Selected_Images_2"

Main_Results_Folder = r"C:\Users\Paula\.spyder-py3\MariaBaron-CS898BA-Project1\Report_CS898BA_Project1\Results_2"

Image_Height = 128
Image_Width = 128

Brightness_Threshold = 70
Contrast_Threshold = 35
Gaussian_Sigma = 1.0

Batch_Size = 32
Epochs = 25
Learning_Rate = 0.001

Validation_Size = 0.20
Test_Size = 0.20

# Read Dataset

Image_Extensions = (".png",".jpg",".jpeg",".ppm")
Dataset_Records = []

Class_Folders = sorted([Folder for Folder in os.listdir(Dataset_Path) if os.path.isdir(os.path.join(Dataset_Path,Folder))])

for Class_Index,Class_Name in enumerate(Class_Folders):
    Class_Path = os.path.join(Dataset_Path,Class_Name)

    for File_Name in sorted(os.listdir(Class_Path)):
        if not File_Name.lower().endswith(Image_Extensions):
            continue

        Image_Path = os.path.join(Class_Path,File_Name)
        Image = cv2.imread(Image_Path)

        if Image is None:
            print("Could not read:",Image_Path)
            continue

        Dataset_Records.append({
            "Image_Path":Image_Path,
            "Class_Name":Class_Name,
            "Class_Index":Class_Index
        })

Dataset_DataFrame = pd.DataFrame(Dataset_Records)

print("Classes:",len(Class_Folders))
print("Images:",len(Dataset_DataFrame))
print("\nImages per class:")
print(Dataset_DataFrame["Class_Name"].value_counts().sort_index())

Train_Data,Test_Data = train_test_split(
    Dataset_DataFrame,
    test_size=Test_Size,
    random_state=Random_Seed,
    stratify=Dataset_DataFrame["Class_Index"]
)

Adjusted_Validation_Size = Validation_Size / (1.0 - Test_Size)

Train_Data,Validation_Data = train_test_split(
    Train_Data,
    test_size=Adjusted_Validation_Size,
    random_state=Random_Seed,
    stratify=Train_Data["Class_Index"]
)

Train_Data = Train_Data.reset_index(drop=True)
Validation_Data = Validation_Data.reset_index(drop=True)
Test_Data = Test_Data.reset_index(drop=True)

# %% Adaptive Preprocessing

Use_Adaptive_Preprocessing = True

if Use_Adaptive_Preprocessing:
    Results_Folder = os.path.join(Main_Results_Folder,"With_Adaptive_Preprocessing")
else:
    Results_Folder = os.path.join(Main_Results_Folder,"Without_Adaptive_Preprocessing")

if os.path.exists(Results_Folder):
    shutil.rmtree(Results_Folder)

os.makedirs(Results_Folder,exist_ok=True)

print("\nUse Adaptive Preprocessing:",Use_Adaptive_Preprocessing)
print("Results folder:",Results_Folder)

Train_Data.to_csv(os.path.join(Results_Folder,"Training_Images.csv"),index=False)
Validation_Data.to_csv(os.path.join(Results_Folder,"Validation_Images.csv"),index=False)
Test_Data.to_csv(os.path.join(Results_Folder,"Testing_Images.csv"),index=False)

print("\nTraining images:",len(Train_Data))
print("Validation images:",len(Validation_Data))
print("Testing images:",len(Test_Data))

Preprocessed_Folder = os.path.join(Results_Folder,"Preprocessed_Test_Images")
os.makedirs(Preprocessed_Folder,exist_ok=True)

def Preprocess_Image(Image_Path,Apply_Adaptive_Preprocessing=False):
    Image_BGR = cv2.imread(Image_Path)

    if Image_BGR is None:
        raise ValueError("Could not read image: " + Image_Path)

    Image_BGR = cv2.resize(Image_BGR,(Image_Width,Image_Height))

    if Apply_Adaptive_Preprocessing:
        
        Gray_Image = cv2.cvtColor(Image_BGR,cv2.COLOR_BGR2GRAY)
        Brightness = np.mean(Gray_Image)
        Contrast = np.std(Gray_Image)

        if Brightness < Brightness_Threshold:
            Image_HSV = cv2.cvtColor(Image_BGR,cv2.COLOR_BGR2HSV)
            H_Channel,S_Channel,V_Channel = cv2.split(Image_HSV)
            V_Channel = cv2.equalizeHist(V_Channel)
            Image_HSV = cv2.merge([H_Channel,S_Channel,V_Channel])
            Image_BGR = cv2.cvtColor(Image_HSV,cv2.COLOR_HSV2BGR)

        Gray_Image = cv2.cvtColor(Image_BGR,cv2.COLOR_BGR2GRAY)
        Contrast = np.std(Gray_Image)

        if Contrast < Contrast_Threshold:
            Image_LAB = cv2.cvtColor(Image_BGR,cv2.COLOR_BGR2LAB)
            L_Channel,A_Channel,B_Channel = cv2.split(Image_LAB)
            CLAHE = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            L_Channel = CLAHE.apply(L_Channel)
            Image_LAB = cv2.merge([L_Channel,A_Channel,B_Channel])
            Image_BGR = cv2.cvtColor(Image_LAB,cv2.COLOR_LAB2BGR)

    Image_BGR = cv2.GaussianBlur(Image_BGR,(3,3),Gaussian_Sigma)
    Image_RGB = cv2.cvtColor(Image_BGR,cv2.COLOR_BGR2RGB)

    return Image_RGB

# Save only the test images after preprocessing

for _,Row in Test_Data.iterrows():
    Image_RGB = Preprocess_Image(
        Row["Image_Path"],
        Apply_Adaptive_Preprocessing=Use_Adaptive_Preprocessing
    )

    Class_Folder = os.path.join(Preprocessed_Folder,Row["Class_Name"])
    os.makedirs(Class_Folder,exist_ok=True)

    Output_Path = os.path.join(Class_Folder,os.path.basename(Row["Image_Path"]))
    Image_BGR = cv2.cvtColor(Image_RGB,cv2.COLOR_RGB2BGR)
    cv2.imwrite(Output_Path,Image_BGR)

print("\nPreprocessed test images saved successfully.")

def Load_Training_Image_TensorFlow(Image_Path,Label):
    Image_Path = Image_Path.numpy().decode("utf-8")
    Image = Preprocess_Image(Image_Path,Apply_Adaptive_Preprocessing=False)
    Image = Image.astype(np.float32) / 255.0

    return Image,np.int32(Label)

def Load_Test_Image_TensorFlow(Image_Path,Label):
    Image_Path = Image_Path.numpy().decode("utf-8")
    Image = Preprocess_Image(
        Image_Path,
        Apply_Adaptive_Preprocessing=Use_Adaptive_Preprocessing
    )
    Image = Image.astype(np.float32) / 255.0

    return Image,np.int32(Label)

def Prepare_Training_Image(Image_Path,Label):
    Image,Label = tf.py_function(
        func=Load_Training_Image_TensorFlow,
        inp=[Image_Path,Label],
        Tout=[tf.float32,tf.int32]
    )

    Image.set_shape([Image_Height,Image_Width,3])
    Label.set_shape([])

    return Image,Label

def Prepare_Test_Image(Image_Path,Label):
    Image,Label = tf.py_function(
        func=Load_Test_Image_TensorFlow,
        inp=[Image_Path,Label],
        Tout=[tf.float32,tf.int32]
    )

    Image.set_shape([Image_Height,Image_Width,3])
    Label.set_shape([])

    return Image,Label

def Create_Dataset(DataFrame,Shuffle=False,Is_Test_Dataset=False,Batch_Size_Value=Batch_Size):
    Dataset = tf.data.Dataset.from_tensor_slices((
        DataFrame["Image_Path"].values,
        DataFrame["Class_Index"].values
    ))

    if Is_Test_Dataset:
        Dataset = Dataset.map(Prepare_Test_Image,num_parallel_calls=tf.data.AUTOTUNE)
    else:
        Dataset = Dataset.map(Prepare_Training_Image,num_parallel_calls=tf.data.AUTOTUNE)

    if Shuffle:
        Dataset = Dataset.shuffle(
            buffer_size=len(DataFrame),
            seed=Random_Seed,
            reshuffle_each_iteration=True
        )

    Dataset = Dataset.batch(Batch_Size_Value)
    Dataset = Dataset.prefetch(tf.data.AUTOTUNE)

    return Dataset

Train_Dataset = Create_Dataset(Train_Data,Shuffle=True,Is_Test_Dataset=False)
Validation_Dataset = Create_Dataset(Validation_Data,Is_Test_Dataset=False)
Test_Dataset = Create_Dataset(Test_Data,Is_Test_Dataset=True)

Number_Of_Classes = len(Class_Folders)

print("\nTraining images:",len(Train_Data))
print("Validation images:",len(Validation_Data))
print("Testing images:",len(Test_Data))
print("Number of classes:",Number_Of_Classes)
print("Adaptive preprocessing applied to test:",Use_Adaptive_Preprocessing)

# %% CNN Classification

tf.keras.backend.clear_session()
tf.random.set_seed(Random_Seed)

Baseline_Model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Image_Height,Image_Width,3)),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(Number_Of_Classes,activation="softmax")
])

Baseline_Model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

Baseline_Model.summary()
Baseline_Model_Path = os.path.join(Results_Folder,"Baseline_CNN.keras")

Baseline_Callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=Baseline_Model_Path,
        monitor="val_loss",
        save_best_only=True
    )
]

Baseline_History = Baseline_Model.fit(
    Train_Dataset,
    validation_data=Validation_Dataset,
    epochs=Epochs,
    callbacks=Baseline_Callbacks
)

# Evaluate Baseline CNN

Baseline_Test_Loss,Baseline_Test_Accuracy = Baseline_Model.evaluate(Test_Dataset)

print("\nBaseline CNN Results")
print("Test Loss:",round(Baseline_Test_Loss,4))
print("Test Accuracy:",round(Baseline_Test_Accuracy,4))
print("Baseline model saved in:",Baseline_Model_Path)

# %% Hyperparameter Optimization

Learning_Rates = [0.01,0.001,0.0001]
Batch_Sizes = [32,64]
Dropout_Rates = [0.3,0.5]

def Create_Optimized_Dataset(DataFrame,Batch_Size_Value,Shuffle=False):
    Dataset = tf.data.Dataset.from_tensor_slices((
        DataFrame["Image_Path"].values,
        DataFrame["Class_Index"].values
    ))

    Dataset = Dataset.map(Prepare_Training_Image,num_parallel_calls=tf.data.AUTOTUNE)

    if Shuffle:
        Dataset = Dataset.shuffle(
            buffer_size=len(DataFrame),
            seed=Random_Seed,
            reshuffle_each_iteration=True
        )

    Dataset = Dataset.batch(Batch_Size_Value)
    Dataset = Dataset.prefetch(tf.data.AUTOTUNE)

    return Dataset

def Build_Model(Dropout_Rate):
    Model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Image_Height,Image_Width,3)),
        tf.keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dropout(Dropout_Rate),
        tf.keras.layers.Dense(Number_Of_Classes,activation="softmax")
    ])

    return Model

Optimization_Results = []
Best_Validation_Accuracy = 0.0
Best_Model_Path = os.path.join(Results_Folder,"Best_Optimized_CNN.keras")
Trial_Number = 1

for Learning_Rate_Value in Learning_Rates:
    for Batch_Size_Value in Batch_Sizes:
        for Dropout_Rate in Dropout_Rates:
            print("\n----------------------------------")
            print("Trial:",Trial_Number)
            print("Learning rate:",Learning_Rate_Value)
            print("Batch size:",Batch_Size_Value)
            print("Dropout rate:",Dropout_Rate)
            print("----------------------------------")

            tf.keras.backend.clear_session()
            tf.random.set_seed(Random_Seed)

            Current_Train_Dataset = Create_Optimized_Dataset(
                Train_Data,
                Batch_Size_Value,
                Shuffle=True
            )

            Current_Validation_Dataset = Create_Optimized_Dataset(
                Validation_Data,
                Batch_Size_Value
            )

            Model = Build_Model(Dropout_Rate)

            Model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate_Value),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            Early_Stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            History = Model.fit(
                Current_Train_Dataset,
                validation_data=Current_Validation_Dataset,
                epochs=Epochs,
                callbacks=[Early_Stopping],
                verbose=1
            )

            Best_Trial_Validation_Accuracy = max(History.history["val_accuracy"])
            Best_Epoch = np.argmax(History.history["val_accuracy"]) + 1

            Optimization_Results.append({
                "Trial":Trial_Number,
                "Learning_Rate":Learning_Rate_Value,
                "Batch_Size":Batch_Size_Value,
                "Dropout_Rate":Dropout_Rate,
                "Best_Epoch":Best_Epoch,
                "Validation_Accuracy":Best_Trial_Validation_Accuracy
            })

            if Best_Trial_Validation_Accuracy > Best_Validation_Accuracy:
                Best_Validation_Accuracy = Best_Trial_Validation_Accuracy
                Model.save(Best_Model_Path)

            Trial_Number += 1

Optimization_DataFrame = pd.DataFrame(Optimization_Results)
Optimization_DataFrame = Optimization_DataFrame.sort_values(
    by="Validation_Accuracy",
    ascending=False
).reset_index(drop=True)

Optimization_CSV_Path = os.path.join(
    Results_Folder,
    "Hyperparameter_Optimization_Results.csv"
)

Optimization_DataFrame.to_csv(Optimization_CSV_Path,index=False)
Best_Parameters = Optimization_DataFrame.iloc[0]

# Save Hyperparameter Optimization Results as Image

Optimization_Display = Optimization_DataFrame.copy()
Optimization_Display["Learning_Rate"] = Optimization_Display["Learning_Rate"].apply(lambda Value:f"{Value:.4f}")
Optimization_Display["Validation_Accuracy"] = Optimization_Display["Validation_Accuracy"].round(4)

Figure,Axis = plt.subplots(figsize=(12,7))
Axis.axis("off")

Table = Axis.table(
    cellText=Optimization_Display.values,
    colLabels=Optimization_Display.columns,
    cellLoc="center",
    loc="center"
)

Table.auto_set_font_size(False)
Table.set_fontsize(8)
Table.scale(1.0,1.45)
Axis.set_title("Hyperparameter Optimization Results",fontsize=14,pad=20)

Hyperparameter_Image_Path = os.path.join(
    Results_Folder,
    "Hyperparameter_Optimization_Results.png"
)

plt.savefig(Hyperparameter_Image_Path,dpi=300,bbox_inches="tight")
plt.show()
plt.close(Figure)

print("\nHyperparameter optimization image saved in:")
print(Hyperparameter_Image_Path)
print("\nBest Hyperparameter Combination")
print(Best_Parameters)
print("\nBest model saved in:")
print(Best_Model_Path)
print("\nOptimization results saved in:")
print(Optimization_CSV_Path)

# %% Evaluate Optimized CNN

Best_Model = tf.keras.models.load_model(Best_Model_Path)
Test_Loss,Test_Accuracy = Best_Model.evaluate(Test_Dataset)

print("\nOptimized CNN Results")
print("Test Loss:",round(Test_Loss,4))
print("Test Accuracy:",round(Test_Accuracy,4))

# Generate Predictions

Prediction_Probabilities = Best_Model.predict(Test_Dataset)
Predicted_Labels = np.argmax(Prediction_Probabilities,axis=1)
True_Labels = Test_Data["Class_Index"].to_numpy()

# Classification Report

Classification_Report = classification_report(
    True_Labels,
    Predicted_Labels,
    target_names=Class_Folders,
    output_dict=True,
    zero_division=0
)

Classification_DataFrame = pd.DataFrame(Classification_Report).transpose()

Classification_Report_Path = os.path.join(
    Results_Folder,
    "Classification_Report.csv"
)

Classification_DataFrame.to_csv(Classification_Report_Path)

# Save Classification Report as Image

Classification_Display = Classification_DataFrame.round(3)

Figure,Axis = plt.subplots(figsize=(10,4))
Axis.axis("off")

Table = Axis.table(
    cellText=Classification_Display.values,
    rowLabels=Classification_Display.index,
    colLabels=Classification_Display.columns,
    cellLoc="center",
    loc="center"
)

Table.auto_set_font_size(False)
Table.set_fontsize(9)
Table.scale(1.2,1.5)
Axis.set_title("Classification Report - Optimized CNN",fontsize=14,pad=20)

Classification_Report_Image = os.path.join(
    Results_Folder,
    "Classification_Report.png"
)

plt.savefig(Classification_Report_Image,dpi=300,bbox_inches="tight")
plt.show()
plt.close(Figure)

print("\nClassification Report:")
print(Classification_DataFrame)
print("\nClassification report image saved in:")
print(Classification_Report_Image)
print("\nClassification report saved in:")
print(Classification_Report_Path)

# Confusion Matrix

Confusion_Matrix = confusion_matrix(True_Labels,Predicted_Labels)

Figure,Axis = plt.subplots(figsize=(10,8))
Matrix_Image = Axis.imshow(
    Confusion_Matrix,
    interpolation="nearest",
    cmap="Blues"
)

Figure.colorbar(Matrix_Image,ax=Axis)

Axis.set_title("Confusion Matrix - Optimized CNN")
Axis.set_xlabel("Predicted Class")
Axis.set_ylabel("True Class")
Axis.set_xticks(np.arange(Number_Of_Classes))
Axis.set_yticks(np.arange(Number_Of_Classes))
Axis.set_xticklabels(Class_Folders,rotation=45,ha="right")
Axis.set_yticklabels(Class_Folders)

Threshold = Confusion_Matrix.max() / 2

for Row_Index in range(Confusion_Matrix.shape[0]):
    for Column_Index in range(Confusion_Matrix.shape[1]):
        Value = Confusion_Matrix[Row_Index,Column_Index]

        Axis.text(
            Column_Index,
            Row_Index,
            Value,
            ha="center",
            va="center",
            color="white" if Value > Threshold else "black"
        )

plt.tight_layout()

Confusion_Matrix_Path = os.path.join(
    Results_Folder,
    "Confusion_Matrix.png"
)

plt.savefig(Confusion_Matrix_Path,dpi=300,bbox_inches="tight")
plt.show()
plt.close(Figure)

print("\nConfusion matrix saved in:")
print(Confusion_Matrix_Path)

# Compare Baseline and Optimized Models

Model_Comparison = pd.DataFrame({
    "Model":["Baseline CNN","Optimized CNN"],
    "Test_Loss":[Baseline_Test_Loss,Test_Loss],
    "Test_Accuracy":[Baseline_Test_Accuracy,Test_Accuracy]
})

Model_Comparison["Test_Loss"] = Model_Comparison["Test_Loss"].round(4)
Model_Comparison["Test_Accuracy"] = Model_Comparison["Test_Accuracy"].round(4)

Comparison_Path = os.path.join(
    Results_Folder,
    "Model_Comparison.csv"
)

Model_Comparison.to_csv(Comparison_Path,index=False)

print("\nModel Comparison:")
print(Model_Comparison)

# Save Model Comparison as Image

Figure,Axis = plt.subplots(figsize=(8,3))
Axis.axis("off")

Table = Axis.table(
    cellText=Model_Comparison.values,
    colLabels=Model_Comparison.columns,
    cellLoc="center",
    loc="center"
)

Table.auto_set_font_size(False)
Table.set_fontsize(10)
Table.scale(1.1,1.6)

Axis.set_title(
    "Baseline and Optimized CNN Comparison",
    fontsize=14,
    pad=20
)

Model_Comparison_Image_Path = os.path.join(
    Results_Folder,
    "Model_Comparison.png"
)

plt.savefig(Model_Comparison_Image_Path,dpi=300,bbox_inches="tight")
plt.show()
plt.close(Figure)

print("\nModel comparison image saved in:")
print(Model_Comparison_Image_Path)
print("\nModel comparison saved in:")
print(Comparison_Path)
print("\nAdaptive preprocessing used on test images:",Use_Adaptive_Preprocessing)
print("\nAll results saved in:")
print(Results_Folder)