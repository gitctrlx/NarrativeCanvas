import numpy as np
import cv2
import tensorrt as trt
from cuda import cudart

class ImageClassifier:
    
    def __init__(self, trt_file, labels_file='imagenet_classes.txt', nHeight=224, nWidth=224):
        self.trt_file = trt_file
        self.labels_file = labels_file
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.class_names = self._read_class_names(self.labels_file)
        self.nHeight = nHeight
        self.nWidth = nWidth
        self.engine = self._load_engine(self.trt_file)
        self.nIO = self.engine.num_io_tensors
        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.nIO)]
        self.nInput = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(trt.TensorIOMode.INPUT)

    def _read_class_names(self, file_path: str):
        try:
            with open(file_path, 'r') as f:
                return [line.strip().split(', ')[1] for line in f.readlines()]
        except FileNotFoundError:
            raise Exception(f"[ERROR]Class labels file '{file_path}' not found.")

    def _load_engine(self, trtFile: str) -> trt.ICudaEngine:
        try:
            with open(trtFile, "rb") as f:
                engine_data = f.read()
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(engine_data)
        except FileNotFoundError:
            raise Exception(f"[ERROR]TensorRT engine file '{trtFile}' not found.")

    def _preprocess_image(self, image_path):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"[ERROR]The image {image_path} could not be found.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.nWidth, self.nHeight), interpolation=cv2.INTER_CUBIC)
        img_resized = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_resized - mean) / std
        return img_normalized.transpose(2, 0, 1)

    def _run_inference(self, data):
        context = self.engine.create_execution_context()
        context.set_input_shape(self.lTensorName[0], data.shape)

        # Initialize host buffers and allocate device memory in one go
        bufferH = [np.ascontiguousarray(data)] + [
            np.empty(context.get_tensor_shape(self.lTensorName[i]), dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i])))
            for i in range(self.nInput, self.nIO)
        ]
        bufferD = [cudart.cudaMalloc(buf.nbytes)[1] for buf in bufferH]

        # Copy input data to device memory and set tensor addresses
        for i in range(self.nIO):
            if i < self.nInput:
                cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            context.set_tensor_address(self.lTensorName[i], int(bufferD[i]))

        # Execute the model asynchronously
        context.execute_async_v3(0)

        # Copy output data from device to host memory
        for i in range(self.nInput, self.nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        return bufferH[1]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict(self, image_paths):
        batch_data = np.stack([self._preprocess_image(img_path) for img_path in image_paths])
        output = self._run_inference(batch_data)
        predictions = []
        for i in range(output.shape[0]):
            probabilities = self.softmax(output[i])
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            label_probs = {self.class_names[j]: float(probabilities[j]) for j in top5_indices}
            predictions.append(label_probs)
        return predictions
    
# Usage
if __name__ == '__main__':
    accuracy='int8'
    model = 'efficientvit-b3-r224'
    image_paths = ['./models/demo/demo.JPEG','./models/demo/demo2.JPEG','./models/demo/demo3.JPEG', './models/demo/demo4.JPEG','./models/demo/demo5.jpg','./models/demo/demo6.png','./models/demo/demo7.JPEG', './models/demo/demo8.JPEG']
    predictions = ImageClassifier(trt_file=f'./models/engine/{model}_{accuracy}.plan', labels_file='./models/imagenet_classes.txt').predict(image_paths)
    print(predictions)
