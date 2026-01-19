import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
import random
import zipfile
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns

# Mount Google Drive and Extract Dataset
drive.mount('/content/drive')
zip_path = "/content/drive/MyDrive/maliiii.zip"
extract_path = "/content/maliiii_extracted"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

dataset_path = os.path.join(extract_path, "Dataset")
train_dir = os.path.join(dataset_path, "Train")
test_dir = os.path.join(dataset_path, "Test")

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(train_dir, target_size=(32, 32), batch_size=16, class_mode='binary')
test_data = datagen.flow_from_directory(test_dir, target_size=(32, 32), batch_size=16, class_mode='binary', shuffle=False)

# Genetic Algorithm Parameters
POPULATION_SIZE = 5
GENERATIONS = 3
MUTATION_RATE = 0.2

# Function to Create Model
def create_model(params):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(params['dense_units'], activation='relu'),
        Dropout(params['dropout']),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to Evaluate Model
def evaluate_model(model):
    history = model.fit(train_data, epochs=2, batch_size=16, validation_data=test_data, verbose=0)
    y_true = test_data.classes
    y_pred_probs = model.predict(test_data)
    y_pred = (y_pred_probs > 0.5).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)
    conf_matrix = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)

    return model, precision, recall, auc, conf_matrix, fpr, tpr, history

# Function to Mutate Hyperparameters
def mutate(params):
    """ Mutates a dictionary of hyperparameters. """
    new_params = params.copy()
    if random.random() < MUTATION_RATE:
        new_params['dense_units'] = random.choice([64, 128, 256])
    if random.random() < MUTATION_RATE:
        new_params['dropout'] = random.uniform(0.2, 0.5)
    if random.random() < MUTATION_RATE:
        new_params['learning_rate'] = random.uniform(0.0001, 0.01)
    return new_params

# Function to Plot Metrics
def plot_metrics(history, conf_matrix, fpr, tpr, auc):
    plt.figure(figsize=(12, 5))

    # Model Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    # Model Loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    # Confusion Matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.show()

    # ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Genetic Algorithm for Hyperparameter Tuning
def genetic_algorithm():
    population = [{
        'dense_units': random.choice([64, 128, 256]),
        'dropout': random.uniform(0.2, 0.5),
        'learning_rate': random.uniform(0.0001, 0.01)}
        for _ in range(POPULATION_SIZE)]

    best_model = None
    best_auc = 0

    for gen in range(GENERATIONS):
        print(f"Generation {gen+1}")
        scores = []
        for params in population:
            model = create_model(params)
            model, precision, recall, auc, conf_matrix, fpr, tpr, history = evaluate_model(model)
            scores.append((auc, model, conf_matrix, fpr, tpr, history))

        scores.sort(reverse=True, key=lambda x: x[0])
        best_model = scores[0][1]
        plot_metrics(scores[0][5], scores[0][2], scores[0][3], scores[0][4], scores[0][0])

        new_population = [params for _, _, _, _, _, _ in scores[:2]]
        while len(new_population) < POPULATION_SIZE:
            new_population.append(mutate(population[0]))  # Fixed mutation input
        population = new_population

    return best_model

# Run Genetic Algorithm and Retrieve the Best Model
best_model = genetic_algorithm()

# Test Cases
def parasite_or_not(x):
    return 'P' if x < 0.5 else 'U'

# Single Prediction Test Case
images, labels = next(test_data)  # Get one batch
print(parasite_or_not(best_model.predict(images)[0][0]))

# Visualization of Test Cases
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(str(parasite_or_not(labels[i])) + " : " +
              str(parasite_or_not(best_model.predict(np.expand_dims(images[i], axis=0))[0][0])))
    plt.axis('off')

plt.show()
