'''

TRANSFER LEARNING

The aim of this program is to build a flower classifier using transfer 
learning on a neural network trained on the ImageNet dataset

'''

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras import Model
import pickle
import matplotlib.pyplot as plt
from os import path

#- - - - - - - - - - - - -Main Functions- - - - - - - - - - - - - - - - -

def task_1():
    
    print('\n===task_1 start===\n')

    print('\nSuccessfully download the small flower dataset from Blackboard and unzip it manually!\n')
    
    print('\n===task_1 end===\n')

    
def task_2():
    '''
    Instantiate a pretrained MobileNetV2 network
    '''
    print('\n===task_2 start===\n')
    
    #Create a base model from MobileNetV2
    build_basemodel()
    
    print('\n===task_2 end===\n')

    
def task_3():
    '''
    Build a model for predicting small_flower_dataset by replacing the last 
    layer of MobileNetV2 with a Dense layer 

    '''
    print('\n===task_3 start===\n')
    
    #load the base model
    model = get_base_model()
    
    #target classes number
    num_class = 5

    #build transfer learning model (replace the last layer)
    outpue_ly_num = -2  
    flr_output = Dense(num_class, activation='softmax')
    flr_output = flr_output(model.layers[outpue_ly_num].output)

    flr_input = model.input
    flr_model = Model(inputs=flr_input, outputs=flr_output)
    
    #freeze all the layer without the last layer
    freeze_ly_num = -1  
    for layer in flr_model.layers[:freeze_ly_num]:
        layer.trainable = False
    
    #save model for other tasks
    flr_model.save("flr_model.h5")

    print('flr_model built and saved successfully!')


    print('\n===task_3 end===\n')

    
def task_4():
    '''
    Prepares training, validation and test sets for the non-accelerated transfer learning
    '''
    print('\n===task_4 start===\n')

    #read the flower data as non-accelerated version training, validation, and test dataset and save them
    precomputed = False
    read_data(precomputed)
    
    print('\n===task_4 end===\n')

    
def task_5():
    '''
    Compile and train model with an SGD optimizer 
    using the following parameters learning_rate=0.01, momentum=0.0, nesterov=False.

    '''
    print('\n===task_5 start===\n')
    
    #load flr_model
    model = get_flr_model()
    
    #parameters setting
    learning_rate = 0.01
    momentum = 0.0
    epochs = 100
    precomputed = False
    plot = False
    
    #complie and train model
    model_training(model, learning_rate, momentum, epochs, precomputed, plot, 'task_5_history')
    
    print('\n===task_5 end===\n')

    
def task_6():
    '''
    Plot the training and validation errors vs time and accuracies vs time from task_5.

    '''
    print('\n===task_6 start===\n')
    
    #load the training history from task_5
    history = pickle.load(open('task_5_history', "rb"))
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    
    #plot error and accuracies
    plot_error_acc(acc, val_acc)

    print('\n===task_6 end===\n')

    
def task_7():
    '''
    Experiment with 3 different learning rate (0.1, 0.005, 0.001). 
    Plot the results, draw conclusions.
    '''
    print('\n===task_7 start===\n')    
    
    #parameters setting
    momentum = 0
    epochs = 200
    precomputed = False
    plot = True
    
    #learning_rate: 0.1, complie and train model, plot result
    model = get_flr_model()
    learning_rate = 0.1    
    print('  learning_rate: ' + str(learning_rate))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    #learning_rate: 0.005, complie and train model, plot result
    model = get_flr_model()
    learning_rate = 0.005  
    print('  learning_rate: ' + str(learning_rate))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    #learning_rate: 0.001, complie and train model, plot result
    model = get_flr_model()
    learning_rate = 0.001   
    print('  learning_rate: ' + str(learning_rate))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    print('\n===task_7 end===\n')

    
def task_8():
    '''
    With the best learning rate (0.005) found from previous task,
    add a non zero momentum to the training with the SGD optimizer 
    (consider 3 values for the momentum).
    '''
    print('\n===task_8 start===\n')  
    
    #parameters setting
    learning_rate = 0.005 
    epochs = 200
    precomputed = False
    plot = True
    
    #momentum: 0.5, complie and train model, plot result
    model = get_flr_model()
    momentum = 0.5   
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
        
    #momentum: 0.9, complie and train model, plot result
    model = get_flr_model()
    momentum = 0.9  
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    #momentum: 0.99, complie and train model, plot result
    model = get_flr_model()
    momentum = 0.99   
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    
    print('\n===task_8 end===\n')

    
def task_9():
    '''
    Prepare accelerated version training, validation and test sets. 
    Those are based on {(F(x1).t1), (F(x2),t2),...,(F(xm),tm)},
    '''
    print('\n===task_9 start===\n')    
    
    #read the flower data as accelerated version training, validation, and test dataset and save them
    precomputed = True
    read_data(precomputed)
    
    print('\n===task_9 end===\n')

    
def task_10():
    '''
    With the best learning rate (0.005) found from previous task,
    add a non zero momentum to the training with the SGD optimizer 
    and use the accelerated version dataset
    (consider 3 values for the momentum).
    '''
    print('\n===task_10 start===\n')    
    
    #parameters setting
    learning_rate = 0.005 
    epochs = 200
    precomputed = True
    plot = True
    
    #momentum: 0.5, complie and train model using accelerated version dataset, plot result
    model = get_accelerated_model()
    momentum = 0.5   
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    #momentum: 0.9, complie and train model using accelerated version dataset, plot result
    model = get_accelerated_model()
    momentum = 0.9  
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    #momentum: 0.99, complie and train model using accelerated version dataset, plot result
    model = get_accelerated_model()
    momentum = 0.99   
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    print('\n===task_10 end===\n')
    
    
#- - - - - - - - - - - - -Testing Functions- - - - - - - - - - - - - - - - -

def test():
    '''
    This function is only for testing and comparing the result from setting 
    different learning rate and momentum
    '''
    print('\n===test start===\n')    
    
    #parameters setting
    epochs = 200
    precomputed = True
    plot = True
    
    #complie and train model using accelerated version dataset, plot result
    model = get_accelerated_model()
    
    learning_rate = 0.1
    momentum = 0.0   
    print('  learning_rate: ' + str(learning_rate))
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    
    model = get_accelerated_model()
    learning_rate = 0.01
    momentum = 0.0   
    print('  learning_rate: ' + str(learning_rate))
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    
    model = get_accelerated_model()
    learning_rate = 0.005
    momentum = 0.0   
    print('  learning_rate: ' + str(learning_rate))
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    
    model = get_accelerated_model()
    learning_rate = 0.001
    momentum = 0.0   
    print('  learning_rate: ' + str(learning_rate))
    print('  momentum: ' + str(momentum))
    model_training(model, learning_rate, momentum, epochs, precomputed, plot)
    
    print('\n===test end===\n')   
    

#- - - - - - - - - - - - -Auxiliary Functions- - - - - - - - - - - - - - - - -
    

def build_basemodel():
    '''
    Load MobileNetV2 as the base model of this transfer learning task.
    save the model as base_model.h5 file

    Returns
    -------
    None.

    '''
    model = keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        alpha=1.0,
        weights="imagenet",
        input_tensor=None,
        classifier_activation="softmax"
    )
    
    model.save("base_model.h5")
    
    print('base_model build and save successfully!')
    
def plot_error_acc(acc, val_acc):
    '''
    Plot accuracies and error of training and validation dataset during training

    Parameters
    ----------
    acc : list
        accuracies history of training dataset during training.
    val_acc : list
        accuracies history of validation dataset during training.

    Returns
    -------
    None.

    '''
    
    #error: 1 - accuracy
    err = [1 - a for a in acc]
    val_err = [1 - a for a in val_acc]
    
    maximum = 1
    
    #subplot 1: error
    nrow = 2
    ncol = 1
    plot_num = 1
    plt.subplot(nrow, ncol, plot_num)
    plt.plot(err, label='Training Error')
    plt.plot(val_err, label='Validation Error')
    plt.ylabel('Error')
    #fix range
    plt.ylim([min(plt.ylim()), maximum])
    plt.legend(['Training Error', 'Validation Error'])
    plt.xlabel('Epoch')
    
    #subplot 2: accuracy
    plot_num = 2
    plt.subplot(nrow, ncol, plot_num)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylabel('Accuracy')
    #fix range
    plt.ylim([min(plt.ylim()), maximum])
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.tight_layout()
    
    plt.show()
    
    print('result show in plots')

    
def format_image(image, label):
    '''
    Rescale and resize image

    Parameters
    ----------
    image : tf.tensor
        The image from dataset.
    label : tf.tensor
        The label of image from dataset.

    Returns
    -------
    image : tf.tensor
        Rescaled and resized image.
    label : tf.tensor
        Same as input.

    '''
    newsize = (160, 160)
    pixel_value = 255

    image = tf.cast(image, tf.float32)
    #rescale the range to [0, 1]
    image = image/pixel_value
    image = tf.image.resize(image, newsize)

    return image, label


def read_data(precomputed):
    '''
    Read small_flower_dataset from specific path and transfer the data 
    into tf.data.Dataset, then save these dataset    

    Parameters
    ----------
    precomputed : Boolean
        Read the data as accelerated version or not.
        if yes: the dataset file name will be like 'precomputed_train_ds'
        if no: the dataset file name will be like 'train_ds'

    Returns
    -------
    None.

    '''
    
    #specific path for loading data
    path = "./small_flower_dataset"
    
    #default image size
    IMG_SIZE = (224, 224)
    
    #read data to dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
      path,
      image_size=IMG_SIZE,
      validation_split=0.3,
      subset='training',
      seed=1,
      batch_size=32)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
      path,
      image_size=IMG_SIZE,
      validation_split=0.3,
      subset='validation',
      seed=1,
      batch_size=32)
    
    #split data into train(70%), validation(~20%), and test(~10%)
    test_val_ratio = 3
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // test_val_ratio)
    val_ds = val_ds.skip(val_batches // test_val_ratio)
    
    file_name = ''
    
    #accelerated version or not
    if precomputed:
        
        file_name = 'precomputed_'

        #prepare model for precomputing
        pretrained_model = get_flr_model()
        output_ly_num = -2
        model = Model(inputs=pretrained_model.inputs, outputs=pretrained_model.layers[output_ly_num].output)
        
        #rescale, resize, and precompute image
        train_ds = train_ds.map(format_image).map(lambda data, l: (model(data, training=False), l))
        val_ds = val_ds.map(format_image).map(lambda data, l: (model(data, training=False), l))
        test_ds = test_ds.map(format_image).map(lambda data, l: (model(data, training=False), l))
        
    else:
        #rescale, resize image
        train_ds = train_ds.map(format_image)
        val_ds = val_ds.map(format_image)
        test_ds = test_ds.map(format_image)
    
    #Use buffered prefetching to load images
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    #save dataset
    train_ds.save(file_name + 'train_ds')
    val_ds.save(file_name + 'val_ds')
    test_ds.save(file_name + 'test_ds')
    
    print('train_ds, val_ds, and test_ds saved successfully!')

    
    
def get_data(precomputed):
    '''
    Get the flower dataset from file, if not exist then call read_data() to 
    produce the dataset

    Parameters
    ----------
    precomputed : Boolean
        get the dataset as accelerated version or not.

    Returns
    -------
    train_ds : tf.data.dataset
        training dataset.
    val_ds : tf.data.dataset
        validation dataset.
    test_ds : tf.data.dataset
        test dataset.

    '''
    
    #determine the file name that going to get
    file_name = 'precomputed_' if precomputed else ''
    
    #check if the dataset exist, if not then call read_data() to produce them
    if path.exists('./' + file_name + 'train_ds')\
        & path.exists('./' + file_name + 'val_ds')\
        & path.exists('./' + file_name + 'test_ds'):
        pass
    else:
        read_data(precomputed)
    
    #load dataset
    train_ds = tf.data.Dataset.load('./' + file_name + 'train_ds')
    val_ds = tf.data.Dataset.load('./' + file_name + 'val_ds')
    test_ds = tf.data.Dataset.load('./' + file_name + 'test_ds')
    
    return train_ds, val_ds, test_ds

def get_base_model():
    '''
    Get the base_model, if not exist then call build_basemodel() to produce it

    Returns
    -------
    model : keras.Model
        the base_model for transfer learning.

    '''
    #check if the model exist, if not then call build_basemodel() to produce it
    if path.exists('./base_model.h5'):
        pass
    else:
        build_basemodel()
    
    #load model
    model = keras.models.load_model('./base_model.h5')
    
    return model

def get_flr_model():
    '''
    Get the flr_model, if not exist then call task_3() to produce it

    Returns
    -------
    model : keras.Model
        the model for predicting small_flower_dataset

    '''
    #check if the model exist, if not then call task_3() to produce it
    if path.exists('./flr_model.h5'):
        pass
    else:
        task_3()
        
    #load model
    model = keras.models.load_model('./flr_model.h5')
    
    return model
    
def model_compile(model, learning_rate, momentum):
    '''
    Compile the input model with the input patameters and SGD optimizer

    Parameters
    ----------
    model : keras.Model
        the model need compile.
    learning_rate : float
        the learning rate when training the model.
    momentum : float
        the momentum when training the model.

    Returns
    -------
    model : keras.Model
        compiled model.

    '''
    
    lr = learning_rate
    m = momentum
    opt = keras.optimizers.SGD(learning_rate=lr, momentum=m, nesterov=False)
    
    #compile model
    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model

def model_training(model, learning_rate, momentum, epochs, precomputed, plot, his_file_name = 'history'):
    '''
    Training the input model with the input parameters and small_flower_dataset.
    Plot the result if need.

    Parameters
    ----------
    model : keras.Model
        the model for predicting small_flower_dataset
    learning_rate : float
        the learning rate when training the model.
    momentum : float
        the momentum when training the model.
    epochs : int
        hoe many epochs need for training the model.
    precomputed : boolean
        get the dataset as accelerated version or not.
    plot : boolean
        need plot the result of not.
    his_file_name : str, optional
        the training history file name. The default is 'history'.

    Returns
    -------
    None.

    '''
    
    lr = learning_rate
    m = momentum
    ech = epochs
    
    #compile
    model = model_compile(model, lr, m)
    
    #get the dataset  
    train_ds, val_ds, test_ds = get_data(precomputed)

    #early stop with 5 epochs with no improvement
    early_stop_number = 5
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_number)

    #training 
    history = model.fit(train_ds, validation_data=val_ds, epochs=ech, callbacks=[callback])
    
    #if need plot or not
    if plot:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plot_error_acc(acc, val_acc)
    #if do not need plot then save the history file
    else:
        with open(his_file_name, 'wb') as file:
                pickle.dump(history.history, file)
    
    #show the result of using test dataset
    loss, accuracy = model.evaluate(test_ds)

    print("test loss: {:.2f}".format(loss))
    print("test accuracy: {:.2f}".format(accuracy))
    
def get_accelerated_model():
    '''
    Build the model for predicting accelerated version small_flower_dataset

    Returns
    -------
    model : keras.Model
        the model for predicting accelerated version small_flower_dataset

    '''
    #target classes number
    num_class = 5
    
    #accelerated version dataset has only 1 dimension
    input_shape = 1280
    inputs = tf.keras.Input(shape=(input_shape))
    outputs = Dense(num_class, activation='softmax')(inputs)
    
    #build model
    model = tf.keras.Model(inputs, outputs)
    
    return model
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
    pass

    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()
    task_7()
    task_8()
    task_9()
    task_10()
    
