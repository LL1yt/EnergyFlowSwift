Converting TensorFlow 2 BERT Transformer Models
The following examples demonstrate converting TensorFlow 2 models to Core ML using Core ML Tools.

Convert the DistilBERT Transformer Model
The following example converts the DistilBERT model from Huggingface to Core ML.

Requirements

This example requires TensorFlow 2 and Transformers version 4.17.0.

Follow these steps:

Add the import statements:

import numpy as np
import coremltools as ct
import tensorflow as tf

from transformers import DistilBertTokenizer, TFDistilBertForMaskedLM
Load the DistilBERT model and tokenizer. This example uses the TFDistilBertForMaskedLM variant:

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
distilbert_model = TFDistilBertForMaskedLM.from_pretrained('distilbert-base-cased')
Describe and set the input layer, and then build the TensorFlow model (tf_model):

max_seq_length = 10
input_shape = (1, max_seq_length) #(batch_size, maximum_sequence_length)

input_layer = tf.keras.layers.Input(shape=input_shape[1:], dtype=tf.int32, name='input')

prediction_model = distilbert_model(input_layer)
tf_model = tf.keras.models.Model(inputs=input_layer, outputs=prediction_model)
Convert the tf_model to an ML program (mlmodel):

mlmodel = ct.convert(tf_model)
Create the input using tokenizer:

# Fill the input with zeros to adhere to input_shape

input_values = np.zeros(input_shape)

# Store the tokens from our sample sentence into the input

input_values[0,:8] = np.array(tokenizer.encode("Hello, my dog is cute")).astype(np.int32)
Use mlmodel for prediction:

mlmodel.predict({'input':input_values}) # 'input' is the name of our input layer from (3)
Convert the TF Hub BERT Transformer Model
The following example converts the BERT model from TensorFlow Hub.

Requirements

This example requires TensorFlow 2, TensorFlow Hub, and Transformers version 4.17.0.

Follow these steps:

Add the import statements:

import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

import coremltools as ct
Describe and set the input layer:

max_seq_length = 384
input_shape = (1, max_seq_length)

input_words = tf.keras.layers.Input(
shape=input_shape[1:], dtype=tf.int32, name='input_words')
input_masks = tf.keras.layers.Input(
shape=input_shape[1:], dtype=tf.int32, name='input_masks')
segment_ids = tf.keras.layers.Input(
shape=input_shape[1:], dtype=tf.int32, name='segment_ids')
Build the TensorFlow model (tf_model):

bert_layer = tf_hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)

pooled_output, sequence_output = bert_layer(
[input_words, input_masks, segment_ids])

tf_model = tf.keras.models.Model(
inputs=[input_words, input_masks, segment_ids],
outputs=[pooled_output, sequence_output])
Convert the tf_model to an ML program:

mlmodel = ct.convert(tf_model, source='TensorFlow')
Define the model.preview.type metadata as "bertqa" so that you can preview the model in Xcode, and then save the model in an mlpackage file:

model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "bertQA"
model.save("BERT_with_preview_type.mlpackage")

Convert TensorFlow 2 BERT Transformer Models
Suggest Edits
The following examples demonstrate converting TensorFlow 2 models to Core ML using coremltools.

Convert the DistilBERT transformer model
The following example converts the DistilBERT model from Huggingface to Core ML.

📘
Install Transformers

You may need to first install Transformers version 2.10.0.

Follow these steps:

Add the import statements:
Python

import numpy as np
import coremltools as ct
import tensorflow as tf

from transformers import DistilBertTokenizer, TFDistilBertForMaskedLM
Load the DistilBERT model and tokenizer. This example uses the TFDistilBertForMaskedLM variant:
Python

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
distilbert_model = TFDistilBertForMaskedLM.from_pretrained('distilbert-base-cased')
Describe and set the input layer, and then build the TensorFlow model (tf_model):
Python

max_seq_length = 10
input_shape = (1, max_seq_length) #(batch_size, maximum_sequence_length)

input_layer = tf.keras.layers.Input(shape=input_shape[1:], dtype=tf.int32, name='input')

prediction_model = distilbert_model(input_layer)
tf_model = tf.keras.models.Model(inputs=input_layer, outputs=prediction_model)
Convert the tf_model to the Core ML format (mlmodel):
Python

mlmodel = ct.convert(tf_model)
Create the input using tokenizer:
Python

# Fill the input with zeros to adhere to input_shape

input_values = np.zeros(input_shape)

# Store the tokens from our sample sentence into the input

input_values[0,:8] = np.array(tokenizer.encode("Hello, my dog is cute")).astype(np.int32)
Use mlmodel for prediction:
Python

mlmodel.predict({'input':input_values}) # 'input' is the name of our input layer from (3)
Convert the TF Hub BERT transformer model
The following example converts the BERT model from TensorFlow Hub. Follow these steps:

Add the import statements:
Python

import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

import coremltools as ct
Describe and set the input layer:
Python

max_seq_length = 384
input_shape = (1, max_seq_length)

input_words = tf.keras.layers.Input(
shape=input_shape[1:], dtype=tf.int32, name='input_words')
input_masks = tf.keras.layers.Input(
shape=input_shape[1:], dtype=tf.int32, name='input_masks')
segment_ids = tf.keras.layers.Input(
shape=input_shape[1:], dtype=tf.int32, name='segment_ids')
Build the TensorFlow model (tf_model):
Python

bert_layer = tf_hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)

pooled_output, sequence_output = bert_layer(
[input_words, input_masks, segment_ids])

tf_model = tf.keras.models.Model(
inputs=[input_words, input_masks, segment_ids],
outputs=[pooled_output, sequence_output])
Convert the tf_model to Core ML (mlmodel):
Python

mlmodel = ct.convert(tf_model, source='TensorFlow')

Load and Convert Model Workflow
The typical conversion process with the Unified Conversion API is to load the model to infer its type, and then use the convert() method to convert it to the Core ML format. Follow these steps:

Import coremltools (as ct for the following code snippets), and load a TensorFlow or PyTorch model.

import coremltools as ct

# Load TensorFlow model

import tensorflow as tf # Tf 2.2.0

tf_model = tf.keras.applications.MobileNet()
import coremltools as ct

# Load PyTorch model (and perform tracing)

torch_model = torchvision.models.mobilenet_v2()
torch_model.eval()

example_input = torch.rand(1, 3, 256, 256)
traced_model = torch.jit.trace(torch_model, example_input)
Convert the TensorFlow or PyTorch model using convert():

# Convert using the same API

model_from_tf = ct.convert(tf_model)

# Convert using the same API. Note that we need to provide "inputs" for pytorch conversion.

model_from_torch = ct.convert(traced_model,
inputs=[ct.TensorType(name="input",
shape=example_input.shape)])
The conversion produces an MLModel object which you can use to make predictions, change metadata, or save to the Core ML format for use in Xcode.

By default, older versions of the Unified Conversion API create a neural network, but you can use the convert_to parameter to specify the mlprogram model type for an ML program model:

# Convert using the same API

model_from_tf = ct.convert(tf_model, convert_to="mlprogram")

# Convert using the same API. Note that we need to provide "inputs" for pytorch conversion.

model_from_torch = ct.convert(traced_model,
convert_to="mlprogram",
inputs=[ct.TensorType(name="input",
shape=example_input.shape)])
Since the neuralnetwork format is widely available, it is still the default format produced by versions of the Unified Conversion API older than 7.0. However, in 7.0 and newer versions, the convert() method produces an mlprogram by default with the iOS15/macOS12 deployment target. You can override this behavior by providing a minimum_deployment_target or convert_to value.

For more information, see the MLModel Overview.

Conversion Options

The convert() method tries to infer as much as possible from the source network, but some information may not be present, such as input names, types, shapes, and classifier options. For more information see Conversion Options.

Convert From TensorFlow 2
TensorFlow 2 models are typically exported as tf.Model objects in the SavedModel or HDF5 file formats. For additional TensorFlow formats you can convert, see TensorFlow 2 Workflow.

The following example demonstrates how to use the convert() method to convert an Xception model from tf.keras.applications:

import coremltools as ct
import tensorflow as tf

# Load from .h5 file

tf_model = tf.keras.applications.Xception(weights="imagenet",
input_shape=(299, 299, 3))

# Convert to Core ML

model = ct.convert(tf_model)
Convert From TensorFlow 1
The conversion API can also convert models from TensorFlow 1. These models are generally exported with the extension .pb, in the frozen protobuf file format, using TensorFlow 1’s freeze graph utility. You can pass this model directly into the convert() method. For details, see TensorFlow 1 Workflow.

The following example demonstrates how to convert a pre-trained MobileNet model in the frozen protobuf format to Core ML.

Download for the Following Example

To run the following example, first download this pre-trained model.

import coremltools as ct

# Convert a frozen graph from TensorFlow 1 to Core ML

mlmodel = ct.convert("mobilenet_v1_1.0_224/frozen_graph.pb")
The MobileNet model in the previous example already has a defined input shape, so you do not need to provide it. However, in some cases the TensorFlow model does not contain a fully defined input shape. You can pass an input shape that is compatible with the model into the convert() method in order to provide the shape information, as shown in the following example.

Download for the Following Example

To run the following example, first download this pre-trained model.

import coremltools as ct

# Needs additional shape information

mlmodel = ct.convert("mobilenet_v2_1.0_224_frozen.pb",
inputs=[ct.TensorType(shape=(1, 224, 224, 3))])
Convert from PyTorch
You can convert PyTorch models that are either traced or in already the TorchScript format. For example, you can convert a model obtained using PyTorch’s save and load APIs to Core ML using the same Unified Conversion API as the previous example:

import coremltools as ct
import torch
import torchvision

# Get a pytorch model and save it as a \*.pt file

model = torchvision.models.mobilenet_v2()
model.eval()
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("torchvision_mobilenet_v2.pt")

# Convert the saved PyTorch model to Core ML

mlmodel = ct.convert("torchvision_mobilenet_v2.pt",
inputs=[ct.TensorType(shape=(1, 3, 224, 224))])
For more details on tracing and scripting to produce PyTorch models for conversion, see Converting from PyTorch.

Set the Compute Units
Normally you convert a model by using convert() without using the compute_units parameter. In most cases you don’t need it, because the converter picks the default optimized path for fast execution while loading the model. The default setting (ComputeUnit.ALL) uses all compute units available, including the Neural Engine (NE), the CPU, and the graphics processing unit (GPU). Whether you are using ML programs or neural networks, the defaults for conversion and prediction are picked to execute the model in the most performant way, as described in Typed Execution.

However, you may find it useful, especially for debugging, to specify the actual compute units when converting or loading a model by using the compute_units parameter. The parameter is based on the MLComputeUnits enumeration in the Swift developer language — compute units are employed when loading a Core ML model, taking in MLmodelConfiguration which includes compute units. Therefore, both the MLModel class and convert() provide the compute_units parameter.

The compute_units parameter can have the following values:

coremltools.ComputeUnit.CPU_ONLY: Limit the model to use only the CPU.

coremltools.ComputeUnit.CPU_AND_GPU: Use both the CPU and GPU, but not the NE.

coremltools.ComputeUnit.CPU_AND_NE: Use both the CPU and NE, but not the GPU.

coremltools.ComputeUnit.ALL: The default setting uses all compute units available, including the NE, CPU, and GPU.

For example, the following converts the model and sets the compute_units to CPU only:

model = ct.convert(tf_model, compute_units=ct.ComputeUnit.CPU_ONLY)

Model Prediction
After converting a source model to a Core ML model, you can evaluate the Core ML model by verifying that the predictions made by the Core ML model match the predictions made by the source model.

The following example makes predictions for the HousePricer.mlmodel using the predict() method.

import coremltools as ct

# Load the model

model = ct.models.MLModel('HousePricer.mlmodel')

# Make predictions

predictions = model.predict({'bedroom': 1.0, 'bath': 1.0, 'size': 1240})
macOS Required for Model Prediction

For the prediction API, coremltools interacts with the Core ML framework which is available on macOS only. The prediction API is not available on Linux.

However, Core ML models can be imported and executed with TVM, which may provide a way to test Core ML models on non-macOS systems.

Types of Inputs and Outputs
Core ML supports several feature types for inputs and outputs. The following are two feature types that are commonly used with neural network models:

ArrayFeatureType, which maps to the MLMultiArray Feature Value in Swift

ImageFeatureType, which maps to the Image Feature Value in Swift

When using the Core ML model in your Xcode app, use an MLFeatureValue, which wraps an underlying value and bundles it with that value’s type, represented by MLFeatureType.

To evaluate a Core ML model in python using the predict() method, use one of the following inputs:

For a multi-array, use a NumPy array.

For an image, use a PIL image python object.

Learn More About Image Input and Output

To learn how to work with images and achieve better performance and more convenience, see Image Input and Output.

Specifying Compute Units
If you don’t specify compute units when converting or loading a model, all compute units available on the device are used for execution including the Neural Engine (NE), the CPU, and the graphics processing unit (GPU).

You can control which compute unit the model runs on by setting the compute_units argument when converting a model (with coremltools.convert()) or loading a model (with coremltools.models.MLModel). Calling predict() on the converted or loaded model restricts the model to use only the specific compute units for execution.

For example, the following sets the compute units to CPU only when loading the model:

model = ct.model.MLModel('path/to/the/saved/model.mlmodel', compute_units=ct.ComputeUnit.CPU_ONLY)
Deprecated Flag

In previous versions of coremltools, you would restrict execution to the CPU by specifying the useCPUOnly=True flag. This flag is now deprecated. Instead, use the compute_units parameter .

For more information and values for this parameter, see Set the Compute Units.

Fast Predictions
A Model can be loaded using the Fast Prediction Optimization Hint. This will prefer the prediction latency at the potential cost of specialization time, memory footprint, and the disk space usage.

model = ct.model.MLModel(
'path/to/the/saved/model.mlmodel',
optimization_hints={ 'specializationStrategy': ct.SpecializationStrategy.FastPrediction }
)
Multi-array Prediction
A model that takes a MultiArray input requires a NumPy array as an input with the predict() call. For example:

import coremltools as ct
import numpy as np

model = ct.models.MLModel('path/to/the/saved/model.mlmodel')

# Print input description to get input shape.

print(model.get_spec().description.input)

input_shape = (...) # insert correct shape of the input

# Call predict.

output_dict = model.predict({'input_name': np.random.rand(\*input_shape)})
Image Prediction
A model that takes an image input requires a PIL image as an input with the predict() call. For example:

import coremltools as ct
import numpy as np
import PIL.Image

# Load a model whose input type is "Image".

model = ct.models.MLModel('path/to/the/saved/model.mlmodel')

Height = 20 # use the correct input image height
Width = 60 # use the correct input image width

# Scenario 1: load an image from disk.

def load_image(path, resize_to=None): # resize_to: (Width, Height)
img = PIL.Image.open(path)
if resize_to is not None:
img = img.resize(resize_to, PIL.Image.ANTIALIAS)
img_np = np.array(img).astype(np.float32)
return img_np, img

# Load the image and resize using PIL utilities.

\_, img = load_image('/path/to/image.jpg', resize_to=(Width, Height))
out_dict = model.predict({'image': img})

# Scenario 2: load an image from a NumPy array.

shape = (Height, Width, 3) # height x width x RGB
data = np.zeros(shape, dtype=np.uint8)

# manipulate NumPy data

pil_img = PIL.Image.fromarray(data)
out_dict = model.predict({'image': pil_img})
Image Prediction for a Multi-array Model
If the Core ML model has a MultiArray input type that actually represents a JPEG image, you can still use the JPEG image for the prediction if you first convert the loaded image to a NumPy array, as shown in this example:

Height = 20 # use the correct input image height
Width = 60 # use the correct input image width

# Assumption: the mlmodel's input is of type MultiArray and of shape (1, 3, Height, Width).

model_expected_input_shape = (1, 3, Height, Width) # depending on the model description, this could be (3, Height, Width)

# Load the model.

model = coremltools.models.MLModel('path/to/the/saved/model.mlmodel')

def load_image_as_numpy_array(path, resize_to=None): # resize_to: (Width, Height)
img = PIL.Image.open(path)
if resize_to is not None:
img = img.resize(resize_to, PIL.Image.ANTIALIAS)
img_np = np.array(img).astype(np.float32) # shape of this numpy array is (Height, Width, 3)
return img_np

# Load the image and resize using PIL utilities.

img_as_np_array = load_image_as_numpy_array('/path/to/image.jpg', resize_to=(Width, Height)) # shape (Height, Width, 3)

# PIL returns an image in the format in which the channel dimension is in the end,

# which is different than Core ML's input format, so that needs to be modified.

img_as_np_array = np.transpose(img_as_np_array, (2,0,1)) # shape (3, Height, Width)

# Add the batch dimension if the model description has it.

img_as_np_array = np.reshape(img_as_np_array, model_expected_input_shape)

# Now call predict.

out_dict = model.predict({'image': img_as_np_array})
Using Compiled Python Models for Prediction
You can use a compiled Core ML model (CompiledMLModel) rather than MLModel for making predictions. For large models, using a compiled model can save considerable time in initializing the model.

For example, Stable Diffusion, adopted by a vibrant community of artists and developers, enables the creation of unprecedented visuals from a text prompt. When using Core ML Stable Diffusion, you can speed up the load time after the initial load by first copying and storing the location of the mlmodelc compiled model to a fixed location, and then initializing the model from that location.

Note

You can’t modify a compiled model like you can an MLModel loaded from a non-compiled mlpackage model file.

Why Use a Compiled Model?
When you initialize a model using (in Python) model=ct.models.MLModel("model.mlpackge"), the Core ML Framework is invoked and the following steps occur, as shown in the following diagram.

Initialize MLModel
This diagram is from Improve Core ML integration with async prediction, presented at the Apple 2023 World Wide Developer Conference.
The mlpackage is compiled into a file with extension mlmodelc . This step is usually very fast.

The compiled model is then instantiated using the specified compute_units captured in the MLModelConfiguration config.

During instantiation, another compilation occurs for backend device specialization, such as for the Neural Engine (NE), which may take a few seconds or even minutes for large models.

This device specialization step creates the final compiled asset ready to be run. This final compiled model is cached so that the expensive device optimization process does not need to run again. The cache entry is linked to the full file system path of the mlmodelc folder.

As you create an MLModel object in Python using an mlpackage, it uses a temporary directory in a new location to place the mlmodelc folder. The mlmodelc file is then deleted after you have made predictions and the Python process has ended.

The next time you start a new Python process and create an MLModel, the compilation to mlmodelc and the subsequent device specialization occurs again. The cached set can’t be used again, because the location of mlmodelc has changed.

By storing the mlmodelc file to a fixed location first, and then initializing the MLModel from that location, you can make sure that the cache model generated remains active for subsequent loads, thereby making them faster. Let’s see how you would do that in code.

Predict From the Compiled Model
To use a compiled model file, follow these steps:

Load a saved MLModel, or convert a model from a training framework (such as TensorFlow or PyTorch).

For instructions on converting a model, see Load and Convert Model Workflow. This example uses the regnet_y_128fg torchvision model and assumes that you have already converted it to a Core ML mlpackage.

Get the compiled model directory by calling its get_compiled_model_path method.

For example, the following code snippet loads a saved MLModel ("regnet_y_128gf.mlpackage") and gets the compiled path:

mlmodel = ct.models.MLModel("regnet_y_128gf.mlpackage")
compiled_model_path = mlmodel.get_compiled_model_path()
The returned directory in compiled_model_path is only temporary. Copy that directory to a new persistent location (as in the following example, regnet_y_128gf with the extension .mlmodelc in the same directory) using the shutil.copytree() method. You can then use CompiledMLModel to load the compiled model from "regnet_y_128gf.mlmodelc":

from shutil import copytree
copytree(compiled_model_path, "regnet_y_128gf.mlmodelc", dirs_exist_ok=True)

mlmodel = ct.models.CompiledMLModel("regnet_y_128gf.mlmodelc")
This step includes compiling for device specialization. Therefore, the first load can still take a long time. However, since the location of the mlmodelc folder is fixed, the cache is able to work, so subsequent calls to model using CompiledMLModel are quick.

For each prediction, use the mlmodel object to take advantage of this caching:

prediction = mlmodel.predict({'x': 2})
With most large models, it should be very quick to use the compiled model again after the first call.

Timing Example
This example demonstrates timing differences with calling a large model. The results are based on running the example on a MacBook Pro M1 Max with macOS Sonoma. Your timing results will vary depending on your system configuration and other factors.

The following code snippet converts a relatively large model from torchvision:

import coremltools as ct
import torchvision
import torch
from shutil import copytree

torch_model = torchvision.models.regnet_y_128gf()
torch_model.eval()
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(torch_model, example_input)

mlmodel = ct.convert(traced_model,
inputs=[ct.TensorType(shape=example_input.shape)],
)

mlmodel.save("regnet_y_128gf.mlpackage")

# save the mlmodelc

compiled_model_path = mlmodel.get_compiled_model_path()
copytree(compiled_model_path, "regnet_y_128gf.mlmodelc", dirs_exist_ok=True)

The following code snippet measures load time:

from time import perf_counter

tick = perf_counter()
mlmodel = ct.models.MLModel("regnet_y_128gf.mlpackage")
print("time taken to load using ct.models.MLModel: {:.1f} secs".format(perf_counter() - tick))

tick = perf_counter()
mlmodel = ct.models.MLModel("regnet_y_128gf.mlpackage")
print("time taken to load using ct.models.MLModel: {:.1f} secs".format(perf_counter() - tick))

tick = perf_counter()
mlmodel = ct.models.CompiledMLModel("regnet_y_128gf.mlmodelc")
print("time taken to load using ct.models.CompiledMLModel: {:.1f} secs".format(perf_counter() - tick))

tick = perf_counter()
mlmodel = ct.models.CompiledMLModel("regnet_y_128gf.mlmodelc")
print("time taken to load using ct.models.CompiledMLModel: {:.1f} secs".format(perf_counter() - tick))
Running the code produces the following output:

time taken to load using ct.models.MLModel: 15.3 secs
time taken to load using ct.models.MLModel: 17.7 secs
time taken to load using ct.models.CompiledMLModel: 14.7 secs
time taken to load using ct.models.CompiledMLModel: 0.1 secs
These results show that it takes relatively the same time to load an MLModel after the first load, while loading a CompiledMLModel takes much less time after the first load.

Typed Execution
A model’s compute precision impacts its performance and numerical accuracy, which may impact the user experience of apps using the model. Core ML models saved as ML programs or neural networks execute with either float 32 or float 16 precision.

This page describes how the precision is determined by the runtime during execution of either type of model. The ability to choose this precision can give you more flexible control over computations and performance.

While its useful to understand how the system works, you do not need a complete understanding of the runtime to get your app working optimally with Core ML. In most cases you do not have to take any special action, either during model conversion with coremltools, or during model prediction in your app with the Core ML framework. The defaults picked by coremltools and the Core ML framework work well with most models and are picked to optimize the performance of your model.

Choosing the Appropriate Precision
Whether you are using neural networks or ML programs, the defaults for conversion and prediction are picked to execute the model in the most performant way. This typically means that portions of your model will run in float 16 precision. This is fine for a majority of machine learning models, since they typically work well with float 16 precision during inference. Higher float 32 precision is usually required only during training.

In some cases the mismatch between the source model trained using float 32 precision and the Core ML model is large enough to significantly affect the user experience. In this case, you would typically want to disable float 16 precision and execute the model with float 32 precision.

If you use convert_to="neuralnetwork" with the convert() method, the precision is tied to the compute unit used for execution, as described in the following section. The CPU guarantees float 32 precision, so you can use the .cpuOnly property in your app’s Swift code when loading the model to enforce the higher precision.

The model format for convert(), by default, is an ML program, which gives you more flexibility. You can set a compute precision of float 32 during the conversion process by using the compute_precision setting, as shown in the following example:

import coremltools as ct

model = ct.convert(source_model,
compute_precision=ct.precision.FLOAT32)
This example produces a float 32 typed Core ML model that executes in float 32 precision irrespective of the value of the MLComputeUnit.

Neural Network Untyped Tensors
A neural network has explicitly typed model inputs and outputs, but does not define the types of intermediate tensors that are produced by one layer and consumed by another. Instead, the intermediate tensors are automatically typed at runtime by the compute unit responsible for executing the layer producing them.

Sample Core ML model description
A sample Core ML model description, as shown in Xcode. The input and output of the model are typed.
Sample Core ML model untyped tensors
A neural network does not define the types of intermediate tensors that are produced by one layer and consumed by another.
The Core ML runtime dynamically partitions the network graph into sections for the Neural Engine (NE), GPU, and CPU, and each unit executes its section of the network using its native type to maximize its performance and the model’s overall performance. The GPU and NE use float 16 precision, and the CPU uses float 32. The execution precision varies based on the hardware and software versions, since the partitioning of the graph varies with hardware and software.

You have some control over numeric precision by configuring the set of allowed compute units when converting the model (such as All, CPU&GPU, or CPUOnly), as shown in following chart:

Control over numeric precision
Controlling numeric precision by configuring the set of allowed compute units when converting the model.
For example, for float 32 precision, the only guaranteed path is to use CPUOnly, which executes entirely on the CPU with float 32 precision. This CPU-only configuration would apply to the model as a whole, and may not provide the best performance.

ML Program Typed Tensors
ML programs describe a neural network in a code style in which all variables in the program are strongly typed. In contrast to a neural network model, the types of all the intermediate tensors of an ML program are specified in the model itself.

ML program typed tensors
In an ML program, the types of all the intermediate tensors are specified in the model itself.
An ML program uses the same automatic partitioning scheme that neural networks use to distribute work to the NE, GPU, and CPU. However, the types of the tensors add an additional constraint. The runtime respects the explicit types as the minimum precision, and will not reduce the precision.

For example, a float 32 typed model will only ever run with float 32 precision. A float 16 typed model may run with float 32 precision as well, depending on the availability of the float 16 version of the op, which in turn may depend on the hardware and software versions. With an ML program, the precision and compute engine are independent of each other.

The ML program runtime supports an expanded set of precisions on the backend engines. All of the ops supported on the GPU runtime are now equally supported in float 16 and float 32 precisions. As a result, a float 32 model saved as an ML program need not be restricted to run only on the CPU. In addition, a few selected ops, such as convolution, are also available with float 16 precision on the CPU, and on newer hardware, it can provide further acceleration.

ML program runtime
The ML program runtime supports an expanded set of precisions on the backend engines.
By default, an ML program produced by the converter is typed in float 16 precision, and run with the default configuration that uses all compute units (similar to the default neural network).

For models sensitive to precision, you can set the precision to float 32 during conversion by using the compute_precision=coremltools.precision.FLOAT32 flag with the convert() method. Such a model is guaranteed to run with float 32 precision on all hardware and software versions. Unlike a neural network, the float 32 typed ML program will run on a CPU as well as the GPU. Only the NE is barred as a compute unit when a model is typed with float 32 precision.

With an ML program, you can even mix and match precision support within a single compute unit, such as the CPU or GPU. If you need float 32 precision for specific layers instead of the entire model, you can selectively preserve float 32 tensor types by using the compute_precision=ct.transform.Float16ComputePrecision() transform during model conversion.

6

I'm trying to write a Swift package that uses a CoreML model. I'm not very familiar with Swift packages creation and I could not make it work. Here is what I've done based on the different posts I've read so far:

Create an empty package
$ mkdir MyPackage
$ cd MyPackage
$ swift package init
$ swift build
$ swift test
Open the Package.swift file with XCode

Drag and drop the MyModel.mlmodel file into the folder Sources/MyPackage

When I click on the MyModel.mlmodel file in XCode, I have the following message displayed under the class name:

Model is not part of any target. Add the model to a target to enable generation of the model class.
Similarly, if I use the command swift build in a Terminal I get the following message:

warning: found 1 file(s) which are unhandled; explicitly declare them as resources or exclude from the target
/Path/To/MyPackage/Sources/MyPackage/MyModel.mlmodel
To solve this, I added MyModel into the target resources in the file Package.swift:
.target(
name: "MyPackage",
dependencies: [],
resources: [.process("MyModel.mlmodel")]),
If I now use the command $ swift build, I don't have the warning anymore and I get the message:

[3/3] Merging module MyPackage
But when I check the MyModel.mlmodel file in XCode, I have the following message displayed under the class name:

Model class has not been generated yet.
To solve this, I used the following command in a Terminal:
$ cd Sources/MyPackage
$ xcrun coremlcompiler generate MyModel.mlmodel --language Swift .
This generated a MyModel.swift file next to the mlmodel file.

I plugged the model in the code MyPackage.swift:
import CoreML

@available(iOS 12.0, \*)
struct MyPackage {
var model = try! MyModel(configuration: MLModelConfiguration())
}
Finally, in the test file MyPackageTests.swift, I create an instance of MyPackage:
import XCTest
@testable import MyPackage

final class MyPackageTests: XCTestCase {
func testExample() {
if #available(iOS 12.0, \*) {
let foo = MyPackage()
} else {
// Fallback on earlier versions
}
}

    static var allTests = [
        ("testExample", testExample),
    ]

}
I get the following error (it seems that the CoreML model was not found):

Thread 1: Fatal error: Unexpectedly found nil while unwrapping an Optional value
I must have missed something... I hope my description was clear and detailed enough. Thank you for your help!

The solution described bellow worked for me. I hope it is correct.

Conversion of the MLModel

The MLModel cannot be used directly in the Swift package. It first needs to be converted.

$ cd /path/to/folder/containg/mlmodel
$ xcrun coremlcompiler compile MyModel.mlmodel .
$ xcrun coremlcompiler generate MyModel.mlmodel . --language Swift
The first xcrun command will compile the model and create a folder named MyModel.mlmodelc. The second xcrun command will generate a MyModel.swift file.

Add the model to the Swift package

We consider that a Swift package already exists and is located in /path/to/MyPackage/.

Copy the MyModel.mlmodelc folder and MyModel.swift file into the folder /path/to/MyPackage/Sources/MyPackage
Add the MyModel.mlmodelc in the target resources in the file Package.swift:
.target(
name: "MyPackage",
dependencies: [],
resources: [.process("MyModel.mlmodelc")]),
Instantiate MyModel

In the Swift code, simply create an instance of MyModel:

let model = try? MyModel(configuration: MLModelConfiguration())
or:

let url = Bundle.module.url(forResource: "MyModel", withExtension: "mlmodelc")!
let model = try? MyModel(contentsOf: url, configuration: MLModelConfiguration())
Troubleshooting

I got a Type 'MLModel' has no member '\_\_loadContents' error at first. This seems to be a bug related to XCode 12. I simply commented the 2 functions that caused a problem.

See here and here for more information.

Share
Improve this answer
Follow
answered Oct 22, 2020 at 9:48
Vincent Garcia's user avatar
Vincent Garcia
86711 gold badge1212 silver badges2323 bronze badges
Sign up to request clarification or add additional context in comments.

1 Comment

ephemer
Over a year ago
Looks good, but you will have issues with resources: [.process(...)] if you have more than one model in your target. Better to use .copy(...) instead.
3

Instead of precompile your model you can compile on the fly:

if let url = Bundle.module.url(forResource: "MyModel", withExtension: "mlmodel") {
let compiledURL = try! MLModel.compileModel(at: url)
let model = try! MLModel(contentsOf: compiledURL, configuration: MLModelConfiguration())
} else if let url = Bundle.module.url(forResource: "MyModel", withExtension: "mlmodelc") {
let model = try! MLModel(contentsOf: url, configuration: MLModelConfiguration())
}
You still need to add resources to your target:

targets: [
.target(
name: "MyPackage",
dependencies: [],
resources: [.process("MyMLFolder")]]
),
.testTarget(
name: "MyPackage Tests",
dependencies: ["MyPackage"]
),
]
Share
Improve this answer
Follow
edited Jul 6, 2022 at 13:06
answered Jul 5, 2022 at 19:17
sciasxp's user avatar
sciasxp
1,2111212 silver badges1212 bronze badges
Comments

2

This line is probably where it goes wrong: var model = try! MyModel(configuration: MLModelConfiguration())

Since you've added a mlmodel file to the package, it hasn't been compiled yet. I'm not an expert on Swift packages, but I don't believe Xcode automatically compiles this model now. You can see this for yourself when you open up the compiled app bundle -- does it have the mlmodel file in it or the mlmodelc folder?

You may need to add the mlmodelc to the package, not the mlmodel. You can create this by doing:

$ xcrun coremlcompiler compile MyModel.mlmodel .
Next, in your app you will need to load the model as follows:

let url = YourBundle.url(forResource: "MyModel", withExtension: "mlmodelc")!
let model = try! MyModel(contentsOf: url, configuration: MLModelConfiguration())
where YourBundle is a reference to the bundle that contains the mlmodelc file (which I guess is the bundle for the Swift package).
