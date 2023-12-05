import os
from glob import glob
from models.Build import Builder
from models.Infer import ImageClassifier


def build_and_test_all_onnx_models(onnx_directory, accuracy):
    onnx_files = glob(os.path.join(onnx_directory, '*.onnx'))
    failed_models = []  # Initialize an empty list to store names of failed models

    # Loop through each ONNX file and build a TensorRT engine
    for onnx_file in onnx_files:
        model_name = os.path.basename(onnx_file).replace('.onnx', '')

        print(f"\n[INFO] Building model for {model_name} with accuracy {accuracy}")
        builder = Builder(onnxFile = onnx_file,  trtFile=f'./models/engine/{model_name}_{accuracy}.plan', 
                      accuracy=accuracy, optimization_level=3,
                      calibrationDataPath='./models/calibdata/',
                      int8cacheFile = f'./models/engine/int8Cache/{model_name}.cache',
                      timingCacheFile = f'./models/engine/timingCache/{model_name}.TimingCache',
                      removePlanCache=False)
        if builder.build_model():
            print(f"[INFO] Model {model_name} built successfully.")
            # Code to test the built model goes here
            image_paths = ['./models/demo/demo.JPEG', './models/demo/demo2.JPEG', './models/demo/demo3.JPEG', './models/demo/demo4.JPEG']
            predictions = ImageClassifier(trt_file = f'./models/engine/{model_name}_{accuracy}.plan', labels_file='./models/imagenet_classes.txt').predict(image_paths)
            print(predictions)
        else:
            print(f"[ERROR] Building model for {model_name} failed.")
            failed_models.append(model_name)  # Add the failed model name to the list

    # After all models have been attempted, print out the failed ones
    if failed_models:
        print(f"[INFO] The following models failed to build: {', '.join(failed_models)}")
    else:
        print("[INFO] All models were built successfully.")

# Define the ONNX and TensorRT directories and the desired accuracy
onnx_directory = './models/onnx'
accuracy = 'fp32'  # Or 'fp32' or 'fp16'

if __name__ == '__main__':
    build_and_test_all_onnx_models(onnx_directory, accuracy)
