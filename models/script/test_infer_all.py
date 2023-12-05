import os
from glob import glob
from Infer import ImageClassifier

def test_engine_models_with_accuracy(engine_directory, image_paths, accuracies):
    test_results = {}  # Dictionary to store test results

    # Loop through each accuracy type
    for accuracy in accuracies:
        # Pattern to match the model files for the current accuracy
        pattern = f'*_{accuracy}.plan'
        engine_files = glob(os.path.join(engine_directory, pattern))

        # Loop through each .plan file that matches the current accuracy and test it
        for engine_file in engine_files:
            model_name_with_accuracy = os.path.basename(engine_file).replace('.plan', '')
            model_name, model_accuracy = model_name_with_accuracy.rsplit('_', 1)
            
            if model_accuracy != accuracy:
                # Skip if the file's accuracy does not match the expected accuracy
                continue
            
            print(f"[INFO] Testing model {model_name_with_accuracy}")
            classifier = ImageClassifier(trt_file=engine_file)
            predictions = classifier.predict(image_paths)
            test_results[model_name_with_accuracy] = predictions
            print(f"Predictions for {model_name_with_accuracy}:")
            print(predictions)

    # Optionally, return the test results if you want to use them later
    return test_results

# Define the engine directory, the paths to the images used for testing, and the list of accuracies
engine_directory = './engine'
image_paths = ['./demo/demo.JPEG', './demo/demo2.JPEG', './demo/demo3.JPEG', './demo/demo4.JPEG']
accuracies = ['int8', 'fp16', 'fp32']

if __name__ == '__main__':
    test_engine_models_with_accuracy(engine_directory, image_paths, accuracies)
