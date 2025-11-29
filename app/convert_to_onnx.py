import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

# 1. Load your trained Keras model
model = tf.keras.models.load_model("best_health_model.keras")

# 2. Make sure model has an output_names attribute (tf2onnx expects it)
if not hasattr(model, "output_names"):
    # Try to build the model by calling it once, if needed
    try:
        dummy_input = np.random.randn(1, model.input_shape[1]).astype("float32")
        _ = model(dummy_input)
    except Exception:
        pass

    # Set a default output name
    setattr(model, "output_names", ["output"])

print("✅ Model loaded. Input shape:", model.input_shape)
print("✅ Model output names:", getattr(model, "output_names", None))

# 3. Define input signature for tf2onnx
spec = (tf.TensorSpec((None, model.input_shape[1]), tf.float32, name="input"),)

# 4. Convert Keras → ONNX
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13  # you can use 13–17, 13 is fine
)

# 5. Save ONNX model
onnx.save(onnx_model, "best_health_model.onnx")

print("🚀 Conversion complete! Saved as best_health_model.onnx")
