from Loader import loadImages
import numpy as np
import csv

def RunPredictions(nn, resultFilename = 'predictions.csv'):
    images, file_ids = loadImages("data/test")
    loaded_test_images = np.stack( images, axis=0)
    test_images = loaded_test_images[:, :, :, 0]
    X_test = test_images.astype('float32')/255
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

    print("Now creating predictions to test data...")
    Y_pred = nn.predict_classes(X_test)

    with open(resultFilename, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(["file_id", "accent"])
        for i in range(len(file_ids)):
            filewriter.writerow([file_ids[i].split(".")[0], Y_pred[i]])

    print("predictions.csv ready")
