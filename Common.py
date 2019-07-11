import matplotlib.pyplot as plt

def ShowHistoryGraphs(history):
    # Using the validation set, we do not need "nn.evaluate(test_images, test_labels)"...
    fig = plt.figure(figsize=(12, 6))
    ax1=plt.subplot(1, 2, 1)

    def plot_loss(history):
        train_loss = history.history['loss']
        test_loss = history.history['val_loss']
        x = list(range(1, len(test_loss) + 1))
        plt.plot(x, test_loss, color = 'red', label = 'test loss')
        plt.plot(x, train_loss, label = 'traning loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.legend()
        
    plot_loss(history) 
    plt.show()

    ax1=plt.subplot(1, 2, 2)

    def plot_accuracy(history):
        train_acc = history.history['acc']
        test_acc = history.history['val_acc']
        x = list(range(1, len(test_acc) + 1))
        plt.plot(x, test_acc, color = 'red', label = 'test accuracy')
        plt.plot(x, train_acc, label = 'training accuracy')  
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epoch')  
        plt.legend(loc='lower right')
        
    plot_accuracy(history)
    plt.show()

def SaveModel(nn, filenameWeights, filenameModelJson):
    # --- save model and weights
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # serialize model to JSON
    nn_json = nn.to_json()
    with open(filenameModelJson, "w") as json_file:
        json_file.write(nn_json)
    # serialize weights to HDF5
    nn.save_weights(filenameWeights)
    print("Saved model to disk")
