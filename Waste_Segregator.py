import numpy as np
import matplotlib.pylab as plt  # pylab- to give MATLAB-like interface in python
from sklearn.model_selection import train_test_split, KFold  # For metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import tensorflow as tf
import tensorflow_hub as hub  # For loading/reusing pre-trained ML models
import tf_keras
from tf_keras import layers
from tf_keras.preprocessing.image import ImageDataGenerator  # For data augmentation

import os
import cv2
import PIL.Image as Image  # For img proc
import pathlib  # For file/data paths, string path to object

# Constants
IMAGE_SHAPE = (224, 224)
classifier_link = 'https://www.kaggle.com/models/google/mobilenet-v3/TensorFlow2/large-075-224-classification/1'
feature_vector_link = 'https://www.kaggle.com/models/google/mobilenet-v3/TensorFlow2/large-075-224-feature-vector/1'

current_dir = os.getcwd()  # Cwd- current working directory
dataset_path = os.path.join(current_dir, 'dataset3')  # Glass- 1936, Metal- 1373, Plastic- 1818
data_dir = pathlib.Path(dataset_path)

X, y = [], []
img_labels = ['glass', 'metal', 'plastic']
num_of_recyclables = 3
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies, fold_losses = [], []
all_y_true, all_y_pred = [], []  # For confusion matrix?

# Data Preprocessing
recyclables_images_dict = {
    'glass': list(data_dir.glob('glass/*')),  # .glob()- to find files/folders that match a given pattern (wildcards like *, ?, etc)
    'metal': list(data_dir.glob('metal/*')), 
    'plastic': list(data_dir.glob('plastic/*')), 
}
recyclables_labels_dict = {
    'glass': 0, 
    'metal': 1, 
    'plastic': 2, 
}

for recyclable_name, images in recyclables_images_dict.items():  # Key, value
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, IMAGE_SHAPE)
        X.append(resized_img)
        y.append(recyclables_labels_dict[recyclable_name])
X, y = np.array(X), np.array(y)  # Process imgs (to satisfy model requirements) then put to np arrays

'''
# Classifier Test, trial
classifier = tf_keras.Sequential([hub.KerasLayer(classifier_link)])  # Model definition using the classifier_link...
test_img = Image.open('glass1.jpg').resize(IMAGE_SHAPE)
test_img = np.array(test_img)/255  # Pillow img to np array, normalize (from [0, 255] to [0, 1])

# Converts test_img shape from (224, 224, 3) to (1, 224, 224, 3) (batch_size, height, width, channels [RGB here]) (the model requirement)
test_img = test_img[np.newaxis, ...]  # np.newaxis- to add new dimension (alias for None when used in array indexing)

img_labels_test = []  # Store img labels (from imagenet-labels.txt)...
with open('imagenet-labels.txt', 'r') as f:
    img_labels_test = f.read().splitlines()
predict = classifier.predict(test_img)  # Results to np array of class probabilities for all possible labels
predicted_label_idx = np.argmax(predict)  # Gets idx of highest probability
predicted_label = img_labels_test[predicted_label_idx]
print(predicted_label)
'''

'''
# Pre-trained Model, transfer learning
pretrained_model_without_top_layer = hub.KerasLayer(
    feature_vector_link, input_shape=(224, 224, 3), trainable=False)  # trainable=false- freezes pretrained model's weights...

# Cross Validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f'Fold {fold+1}')
    
    # Split data into train and test for current fold
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    X_train_fold, X_test_fold, y_train_fold, y_test_fold = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    X_train_fold_scaled, X_test_fold_scaled = X_train_fold/255, X_test_fold/255  # Normalize

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train_fold_scaled)
    
    # Model Architecture Definition
    model = tf_keras.Sequential([
        pretrained_model_without_top_layer, 
        # 50% Dropout (Regularization, reduces overfitting)- switch modes? (only active in training [automatically disabled in .evaluate() / .predict()]
        layers.Dropout(0.5), 
        layers.Dense(128, activation='relu'),  # Additional Dense Layer (increases capacity, try Leaky Relu?)- can be removed as might cause overfitting
        layers.BatchNormalization(),  # Batch Normalization- improves training speed, stability, and convergence
        # L2 (Regularization, reduces overfitting)- to penalize large weights (like weight decay?)
        layers.Dense(num_of_recyclables, activation='softmax', kernel_regularizer=tf_keras.regularizers.l2(0.01))
        # Check layer placements & adjust values in dense/dropout/l2...
        # Overfitting clues- Accuracy (high in training but low in validation), Loss (low in training but high in validation), poor perf on unseen data, 
    ])
    # print(model.summary())

    # Model Compilation
    model.compile(
        optimizer='adam',  # adam- adaptive lr, try SGD/etc
        # SparseCCE- for multi-class (integer labels, vs CCE- for one-hot labels), from_logits=False (since last layer is softmax)
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy']  # Also compute for accuracy (score[1], so we'll get both loss and accuracy)
    )

    # Model Training
    model.fit(datagen.flow(X_train_fold_scaled, y_train_fold, batch_size=32),  # For the data augmentation
            validation_data=(X_test_fold_scaled, y_test_fold),
            epochs=10)

    # Model Evaluation
    score = model.evaluate(X_test_fold_scaled, y_test_fold)
    fold_accuracies.append(score[1])
    fold_losses.append(score[0])
    
    # Confusion Matrix and F1 Score
    y_pred_fold = model.predict(X_test_fold_scaled)
    y_pred_class = np.argmax(y_pred_fold, axis=1)
    all_y_true.extend(y_test_fold)
    all_y_pred.extend(y_pred_class)
    # conf_matrix = confusion_matrix(y_test_fold, y_pred_class)
    f1 = f1_score(y_test_fold, y_pred_class, average='weighted')
    print(f'F1 Score for Fold {fold + 1}: {f1}')
    conf_matrix = confusion_matrix(y_test_fold, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=img_labels)
    disp.plot(cmap=plt.cm.Blues)  # plt.cm- a colormap module in matplotlib, Blues- sequential colormap from light blue (low values) to dark (high)
    plt.title(f'Confusion Matrix for Fold {fold + 1}')
    plt.show(block=False)  # Non-blocking execution (won't stop execution, we instead wait for 1s then close window)
    plt.pause(1)  # Pause for 1s then 
    
# After Cross Validation
mean_accuracy, std_accuracy = np.mean(fold_accuracies), np.std(fold_accuracies)
quantile_accuracy = np.quantile(fold_accuracies, [0.025, 0.975])  # 95%
mean_loss, std_loss = np.mean(fold_losses), np.std(fold_losses)
quantile_loss = np.quantile(fold_losses, [0.025, 0.975])
print('Cross Validation Accuracies' + str(fold_accuracies))
print('Cross Validation Results- ' + str(mean_accuracy) + ', ' + str(std_accuracy) + ', ' + str(quantile_accuracy))
print('Cross Validation Losses' + str(fold_losses))
print('Cross Validation Results- ' + str(mean_loss) + ', ' + str(std_loss) + ', ' + str(quantile_loss))

overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
print(f'Overall F1 Score: {overall_f1}')

overall_conf_matrix = confusion_matrix(all_y_true, all_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=overall_conf_matrix, display_labels=img_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Overall Confusion Matrix')
plt.show(block=False)
plt.pause(1)

model.save(os.path.join(current_dir, 'Waste_Segregator_Model.h5'))
model.save_weights(os.path.join(current_dir, 'Waste_Segregator_Model_Weights.h5'))

# Try finetuning (Hyperparameter Tuning, lr, momentum, weight decay), add personal data, other algorithms (ResNet), other evaluators (ROC AUC)
'''

###
# Import Model
model_path = os.path.join(current_dir, 'Waste_Segregator_Model.h5')
model = tf_keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# Capture Image
def preprocess_img(img):
    img = cv2.resize(img, IMAGE_SHAPE)
    img = img/255
    img = np.expand_dims(img, axis=0)  # Add batch dimension- for model requirement...
    return img

# Setup Live Webcam
cap = cv2.VideoCapture(2)  # 
print("Space- Capture, Q- Quit")

while True:
    ret, frame = cap.read()  # Read frame from the webcam
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display live webcam feed
    cv2.imshow("Webcam", frame)
    
    # Wait for user to press the key
    key = cv2.waitKey(1)  # Wait 1s then continues if no key was pressed
    # key = cv2.waitKey(0)
    if key == ord(' '):  # If Spacebar
        # Call preprocess_img()
        preprocessed_img = preprocess_img(frame)
        
        # Predict
        prediction = model.predict(preprocessed_img)
        predicted_label_idx = np.argmax(prediction)
        predicted_label = img_labels[predicted_label_idx]
        print(predicted_label)
        
        # Display predicted label on frame
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # <>, <>, (<px from left>, <px from right>), <>, <font scale>, <txt color in BGR> (green), <thickness of lines to draw chars>
        
        # Show the frame with the predicted label
        cv2.imshow("Webcam", frame)
    
    elif key == ord('q'):  # If Q
        # break
        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        break
###

# # Import Model
# IMAGE_SHAPE = (224, 224)
# img_labels = ['glass', 'metal', 'plastic']
# model_path = 'C:/Users/danie/Desktop/Codes/Python/Practice/Image Categorizer/my_model.h5'
# model = tf_keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# test_img = Image.open('vcut.jpg').resize(IMAGE_SHAPE)
# test_img = np.array(test_img)/255
# test_img = np.expand_dims(test_img, axis=0)

# predict = model.predict(test_img)
# predicted_label_idx = np.argmax(predict)
# predicted_label = img_labels[predicted_label_idx]

# print(predicted_label)


