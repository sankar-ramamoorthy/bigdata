import os
import sys
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
#os.environ["SPARK_HOME"] = "/content/spark-3.0.1-bin-hadoop3.2"
#os.environ["_JAVA_OPTIONS"]="-Xms512m -Xmx1024m  -Xss512m -XX:MaxPermSize=1024m"
os.environ["_JAVA_OPTIONS"]="-Xms512m -Xmx4096m  -Xss512m "
#os.environ['HADOOP_HOME'] = "/content/hadoop-3.3.0"
#sys.path.append("/content/hadoop-3.3.0/bin")




#import findspark
#findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import *
from functools import reduce

from pyspark.sql.functions import col

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit, col
from pyspark.sql.types import *
from functools import reduce
from pyspark.sql.functions import col
from petastorm.spark import SparkDatasetConverter, make_spark_converter
import numpy as np
import torch
import torchvision
from PIL import Image
from functools import partial
from petastorm import TransformSpec
from torchvision import transforms
import errno
from datetime import datetime
import shutil
import random
import matplotlib.pyplot as charts
import time
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm.spark_utils import dataset_as_rdd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

spark = SparkSession.builder.config("spark.driver.memory", "12g").config("spark.executor.memory", "12g").getOrCreate()
#spark = SparkSession.builder.config("spark.executor.memory", "6g").getOrCreate()
sc = spark.sparkContext

train_covid_df =  spark.read.format("image").option("dropInvalid", True).load("hdfs://mnt/data/X-RayImageDataSet/train/covid")
train_covid_df = train_covid_df.withColumn("class_name", lit(1))
train_covid_df = train_covid_df.withColumn("class_name", train_covid_df["class_name"].cast(LongType()))
train_covid_df = train_covid_df.withColumn("width", col("image").width)
train_covid_df = train_covid_df.withColumn("height", col("image").height)
train_covid_df = train_covid_df.withColumn("image", col("image").data)

train_nofindings_df =  spark.read.format("image").option("dropInvalid", True).load("/mnt/data/X-RayImageDataSet/train/nofindings")
train_nofindings_df = train_nofindings_df.withColumn("class_name", lit(0))
train_nofindings_df = train_nofindings_df.withColumn("class_name", train_nofindings_df["class_name"].cast(LongType()))
train_nofindings_df = train_nofindings_df.withColumn("width", col("image").width)
train_nofindings_df = train_nofindings_df.withColumn("height", col("image").height)
train_nofindings_df = train_nofindings_df.withColumn("image", col("image").data)

train_pneumonia_df =  spark.read.format("image").option("dropInvalid", True).load("/mnt/data/X-RayImageDataSet/train/pneumonia")
train_pneumonia_df = train_pneumonia_df.withColumn("class_name", lit(2))
train_pneumonia_df = train_pneumonia_df.withColumn("class_name", train_pneumonia_df["class_name"].cast(LongType()))
train_pneumonia_df = train_pneumonia_df.withColumn("width", col("image").width)
train_pneumonia_df = train_pneumonia_df.withColumn("height", col("image").height)
train_pneumonia_df = train_pneumonia_df.withColumn("image", col("image").data)

train_df = train_covid_df.union(train_nofindings_df).union(train_pneumonia_df)

test_covid_df =  spark.read.format("image").option("dropInvalid", True).load("/mnt/data/X-RayImageDataSet/test/covid")
test_covid_df = test_covid_df.withColumn("class_name", lit(1))
test_covid_df = test_covid_df.withColumn("class_name", test_covid_df["class_name"].cast(LongType()))
test_covid_df = test_covid_df.withColumn("width", col("image").width)
test_covid_df = test_covid_df.withColumn("height", col("image").height)
test_covid_df = test_covid_df.withColumn("image", col("image").data)

test_nofindings_df =  spark.read.format("image").option("dropInvalid", True).load("/mnt/data/X-RayImageDataSet/test/nofindings")
test_nofindings_df = test_nofindings_df.withColumn("class_name", lit(0))
test_nofindings_df = test_nofindings_df.withColumn("class_name", test_nofindings_df["class_name"].cast(LongType()))
test_nofindings_df = test_nofindings_df.withColumn("width", col("image").width)
test_nofindings_df = test_nofindings_df.withColumn("height", col("image").height)
test_nofindings_df = test_nofindings_df.withColumn("image", col("image").data)

test_pneumonia_df =  spark.read.format("image").option("dropInvalid", True).load("/mnt/data/X-RayImageDataSet/test/pneumonia")
test_pneumonia_df = test_pneumonia_df.withColumn("class_name", lit(2))
test_pneumonia_df = test_pneumonia_df.withColumn("class_name", test_pneumonia_df["class_name"].cast(LongType()))
test_pneumonia_df = test_pneumonia_df.withColumn("width", col("image").width)
test_pneumonia_df = test_pneumonia_df.withColumn("height", col("image").height)
test_pneumonia_df = test_pneumonia_df.withColumn("image", col("image").data)

test_df = test_covid_df.union(test_pneumonia_df).union(test_nofindings_df)

train_df = train_df.repartition(100)
test_df = test_df.repartition(100)

train_df.printSchema()
test_df.printSchema()

from pyspark.sql.functions import col

from petastorm.spark import SparkDatasetConverter, make_spark_converter

import io
import numpy as np
import torch
import torchvision
from PIL import Image
from functools import partial
from petastorm import TransformSpec
from torchvision import transforms

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///mnt/data/petacache")

converted_train_df = make_spark_converter(train_df)

converted_test_df = make_spark_converter(test_df)

print(f"train_covid: {len(converted_train_df)}")
print(f"test_covid: {len(converted_test_df)}")

#Function to show progress bar in the results
def ShowProgress(total, progress, label):
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r "+label+" [{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)

    sys.stdout.write(text)
    sys.stdout.flush()


import os
import shutil
import random
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as charts

from petastorm.spark import SparkDatasetConverter, make_spark_converter

from petastorm.spark_utils import dataset_as_rdd
from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)

print('Using PyTorch version', torch.__version__)

#Transformation function for each of the row(image) in the dataset
def transform_row(is_train, pd_batch):
  #Load grascale(mode=L) image and convert to 3 channel RGB. This is required by all pretrained models.
  transformers = [transforms.Lambda(lambda x: Image.frombytes(mode="L", size=(x[0], x[1]), data=x[2]).convert('RGB'))]

  #Transformation on image
  if is_train:
    transformers.extend([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.RandomHorizontalFlip(),
    ])
  else:
    transformers.extend([
      transforms.Resize(256),
      transforms.CenterCrop(224),
    ])

  transformers.extend([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #verify if this works for models other than resnet
  ])

  trans = transforms.Compose(transformers)

  pd_batch['image_meta'] = pd_batch.apply(lambda row: [row.width, row.height, row.image], axis=1)

  #Drop unwanted columns and keep only features and class_names columns
  pd_batch['features'] = pd_batch['image_meta'].map(lambda x: trans(x).numpy())#feature extraction
  pd_batch = pd_batch.drop(labels=['image'], axis=1)
  pd_batch = pd_batch.drop(labels=['width'], axis=1)
  pd_batch = pd_batch.drop(labels=['height'], axis=1)
  pd_batch = pd_batch.drop(labels=['image_meta'], axis=1)
  return pd_batch

#Trandformation function for the whole dataset
def get_transform_spec(is_train=True):
  return TransformSpec(partial(transform_row, is_train),
                       edit_fields=[('features', np.float32, (3, 224, 224), False)],
                       selected_fields=['features', 'class_name'])

CLASS_NAMES = ["nofindings","covid","pneumonia"]

NUM_CLASSES = 3
result_time_stamp = None

#Function to create results directory
def CreateResultDirectory(experiment_id):
    mydir = os.path.join(
        os.getcwd(),
        "drive/MyDrive/covid19/results",
        result_time_stamp,
        experiment_id)
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return mydir

#Function to write results to the results directory
def WriteToFile(list, directory, filename):
  with open(os.path.join(directory, filename), 'w') as d:
        d.writelines(list)

#Function to display images. Not used
def show_images(images, labels, preds, BATCH_SIZE):
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):
        plt.subplot(1, BATCH_SIZE, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'

        plt.xlabel(f'{CLASS_NAMES[int(labels[i].numpy())]}')
        plt.ylabel(f'{CLASS_NAMES[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

#Function to plot accuracy and loss curves
def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, MODEL_NAME, resultDirectory):
    graph, labels = plt.subplots(1, 2, figsize=(20, 10))
    labels[0].set_title('Loss Curves - ' + MODEL_NAME)
    labels[0].plot(train_losses, 'C0', label='Training Loss')
    labels[0].plot(valid_losses, 'C1', label='Validation Loss')
    labels[0].legend(loc="upper right")
    labels[0].set_xlabel("Epoch")
    labels[0].set_ylabel("Loss")

    labels[1].set_title('Accuracy Curves -  ' + MODEL_NAME)
    labels[1].plot(train_accuracies, 'C0', label='Training Accuracy')
    labels[1].plot(valid_accuracies, 'C1', label='Validation Accuracy')
    labels[1].legend(loc="upper left")
    labels[1].set_xlabel("Epoch")
    labels[1].set_ylabel("Accuracy")

    graph.savefig(resultDirectory + '/Learning_Curve-'+ MODEL_NAME +'.png')


def plot_confusion_matrix(results, MODEL_NAME, resultDirectory):
    y_true, y_pred = zip(*results)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.astype('float').sum(axis=1)[:, np.newaxis]
    cm = np.around(cm, decimals=2)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix -  ' + MODEL_NAME)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    plt.tight_layout()

    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                    color = 'black'
                    if cm[i,j] > threshold:
                            color = 'white'

                    plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

    plt.savefig(resultDirectory + '/cm-'+MODEL_NAME+'.png')
#BATCH_SIZE = 6

#device = torch.device("cpu")
##resnet18 = torchvision.models.resnet18(pretrained=True)
#resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
#loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)

#Function to set requires grad to False
def set_parameter_requires_grad(model):
  for param in model.parameters():
      param.requires_grad = False

#Function to initialie different models
def initialize_model(MODEL_NAME):
  model = None
  if MODEL_NAME == "resnet18":
    model = torchvision.models.resnet18(pretrained=True)
    set_parameter_requires_grad(model)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
  elif MODEL_NAME == "resnet34":
    model = torchvision.models.resnet34(pretrained=True)
    set_parameter_requires_grad(model)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
  elif MODEL_NAME == "resnet50":
    model = torchvision.models.resnet50(pretrained=True)
    set_parameter_requires_grad(model)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
  elif MODEL_NAME == "resnet101":
    model = torchvision.models.resnet101(pretrained=True)
    set_parameter_requires_grad(model)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
  elif MODEL_NAME == "densenet":
    model = torchvision.models.densenet121(pretrained=True)
    set_parameter_requires_grad(model)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
  elif MODEL_NAME == "alexnet":
    model = torchvision.models.alexnet(pretrained=True)
    set_parameter_requires_grad(model)
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
  elif MODEL_NAME == "vgg":
    model = torchvision.models.vgg11_bn(pretrained=True)
    set_parameter_requires_grad(model)
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
  elif MODEL_NAME == "squeezenet":
    model = torchvision.models.squeezenet1_0(pretrained=True)
    set_parameter_requires_grad(model)
    model.classifier[1] = torch.nn.Conv2d(512, NUM_CLASSES, kernel_size=(1,1), stride=(1,1))
    model.num_classes = NUM_CLASSES
  else:
    print("Invalid model name, exiting...")
    exit()


  return model

def show_preds(images, labels):
    resnet18.eval()
    outputs = resnet18(images)
    _, preds = torch.max(outputs, 1)
    show_images(images, labels, preds)

#Function to train models
def train_model(model, loss_fn, optimizer, device, dataloader_iter, steps_per_phase, source_data_length, is_train=True):
    loss = 0.
    accuracy = 0.
    results = []
    progressText = "Training"

    if is_train == True:
      model.train()
    else:
      model.eval()
      progressText = "Validating"

    for step in range(steps_per_phase):
        ShowProgress(steps_per_phase, step+1, progressText)

        pb_batch = next(dataloader_iter)
        images, labels = pb_batch['features'].to(device), pb_batch['class_name'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        lossfn = loss_fn(outputs, labels)

        if is_train == True:
          lossfn.backward()
          optimizer.step()

        loss += lossfn.item()

        _, preds = torch.max(outputs, 1)
        accuracy += sum((preds.cpu() == labels.cpu()).numpy())

        if is_train == False:
          y_true = labels.detach().to('cpu').numpy().tolist()
          y_pred = preds.detach().to('cpu').numpy().tolist()
          results.extend(list(zip(y_true, y_pred)))

    loss /= steps_per_phase
    accuracy /= source_data_length
    return loss,accuracy,results

#Function to run model based on num of epocs
def Run_Model(MODEL_NAME, BATCH_SIZE, NUM_EPOCHS, LR, MOM, resultDirectory):

  device=None
  model = initialize_model(MODEL_NAME)
  bestModel = None
  best_val_acc = 0.0

  if torch.cuda.is_available():
    device = torch.device("cuda")
    model.cuda()
  else:
    device = torch.device("cpu")
    model.cpu()

  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

  train_losses, train_accuracies = [], []
  valid_losses, valid_accuracies = [], []
  log_result=[]


  with converted_train_df.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), batch_size=BATCH_SIZE) as train_dataloader, \
      converted_test_df.make_torch_dataloader(transform_spec=get_transform_spec(is_train=False), batch_size=BATCH_SIZE) as test_dataloader:

    train_dataloader_iter = iter(train_dataloader)
    steps_per_training = len(converted_train_df) // BATCH_SIZE

    test_dataloader_iter = iter(test_dataloader)
    steps_per_validation = len(converted_test_df) // BATCH_SIZE


    for e in range(NUM_EPOCHS):
        print("\n")
        print(f'Epoch: {e+1}/{NUM_EPOCHS}')

        train_loss, train_accuracy, results = train_model(model, loss_fn, optimizer, device, train_dataloader_iter, steps_per_training, len(converted_train_df), is_train=True)
        valid_loss, valid_accuracy, results = train_model(model, loss_fn, optimizer, device, test_dataloader_iter, steps_per_validation,  len(converted_test_df), is_train=False)

        if(scheduler!=None):
          scheduler.step()

        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}', f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {valid_accuracy:.4f}')
        log_result.append(f'Epoch: {e+1}/{NUM_EPOCHS}' + f' - Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}' + f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {valid_accuracy:.4f}' + '\n')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        is_best = valid_accuracy > best_val_acc
        if is_best:
          best_val_acc = valid_accuracy
          bestModel = model

    best_model_path = resultDirectory + '/best-'+ str(best_val_acc) + "-" +MODEL_NAME+'-model.pth'
    torch.save(bestModel, best_model_path)
    print("\n")
    print("Generating plots and confusion matrix:")
    best_model = torch.load(best_model_path)
    test_loss, test_accuracy, test_results = train_model(best_model, loss_fn, optimizer, device, test_dataloader_iter, steps_per_validation,  len(converted_test_df), is_train=False)
    plot_confusion_matrix(test_results, MODEL_NAME, resultDirectory)

    WriteToFile(log_result, resultDirectory, "Result_Log_"+MODEL_NAME+".txt")
    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, MODEL_NAME, resultDirectory)

def Run_Experiment(experiment_id, models_to_evaluate, BATCH_SIZE, NUM_EPOCHS, LR, MOM):
  resultDirectory = CreateResultDirectory("Experiment " + str(experiment_id))
  print("Results folder: " + resultDirectory)

  WriteToFile(["Batch Size: " + str(BATCH_SIZE) + "\n","Num Epochs: " + str(NUM_EPOCHS) + "\n", "Learning Rate: " + str(LR) + "\n", "Momentum: " + str(MOM) + "\n"], resultDirectory, "Result_Metadata.txt")

  for model_nm in models_to_evaluate:
    print("Running Experiment "+ str(experiment_id) +" for model: " + model_nm)
    Run_Model(model_nm, BATCH_SIZE, NUM_EPOCHS, LR, MOM, resultDirectory)


def test(epochs, test_dataloader_iter, steps_per_validation):
    print('='*20)
    print(f'Starting epoch {epochs}')
    print('='*20)
    val_loss = 0.
    accuracy = 0.

    resnet18.eval() # set model to eval phase

    for test_step in range(steps_per_validation):
        if test_step % 20 == 0:
            percent = test_step/steps_per_validation*100
            print(f'PercentComplete: {percent:.2f}')

        pb_batch = next(test_dataloader_iter)
        images, labels = pb_batch['features'].to(device), pb_batch['class_name'].to(device)
        optimizer.zero_grad()
        outputs = resnet18(images)
        loss = loss_fn(outputs, labels)
        val_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        accuracy += sum((preds == labels).numpy())

    val_loss /= (test_step + 1)
    accuracy = accuracy/len(converted_test_df)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

    return val_loss,accuracy


result_time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


models_to_evaluate = ["resnet18","densenet", "vgg", "squeezenet"]
Run_Experiment(1,models_to_evaluate, BATCH_SIZE=32, NUM_EPOCHS=20, LR=0.001, MOM=0.9)

models_to_evaluate = ["resnet18","densenet", "vgg", "squeezenet"]
Run_Experiment(2,models_to_evaluate, BATCH_SIZE=32, NUM_EPOCHS=20, LR=0.0001, MOM=0.9)

models_to_evaluate = ["resnet18","densenet", "vgg", "squeezenet"]
Run_Experiment(3,models_to_evaluate, BATCH_SIZE=32, NUM_EPOCHS=20, LR=0.00001, MOM=0.9)
