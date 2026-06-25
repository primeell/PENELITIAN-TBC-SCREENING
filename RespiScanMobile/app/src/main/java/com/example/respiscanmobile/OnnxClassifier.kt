package com.example.respiscanmobile

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.InputStream
import java.nio.FloatBuffer
import kotlin.math.exp

class OnnxClassifier(private val context: Context) {
    companion object {
        const val DECISION_THRESHOLD = 0.3595f
        private const val MODEL_NAME = "best_RespiScan_FocalLoss.onnx"
    }

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    init {
        loadModel()
    }

    private fun loadModel() {
        var inputStream: InputStream? = null
        try {
            inputStream = context.assets.open(MODEL_NAME)
            val modelBytes = inputStream.readBytes()
            ortSession = ortEnv.createSession(modelBytes)
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            inputStream?.close()
        }
    }

    fun classifyCough(flatTensor: FloatArray): Float {
        val session = ortSession ?: throw IllegalStateException("ONNX model is not loaded!")
        
        // 1. Prepare input tensor shape: [batch_size = 1, channels = 3, height = 224, width = 224]
        val shape = longArrayOf(1, 3, 224, 224)
        
        // 2. Wrap flat float array into a direct FloatBuffer
        val floatBuffer = FloatBuffer.wrap(flatTensor)
        
        // 3. Create input OnnxTensor
        val inputName = session.inputNames.iterator().next()
        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)
        
        // 4. Run inference
        val outputs = session.run(mapOf(inputName to inputTensor))
        
        // 5. Extract results
        val outputTensor = outputs[0].value as OnnxTensor
        val logits2D = outputTensor.value as Array<FloatArray> // shape [1][2]
        val logits = logits2D[0] // shape [2]
        
        // 6. Compute softmax probability for class 1 (TB)
        val exp0 = exp(logits[0].toDouble())
        val exp1 = exp(logits[1].toDouble())
        val sumExp = exp0 + exp1
        
        val probTb = (exp1 / sumExp).toFloat()
        
        // Clean up native tensors
        inputTensor.close()
        outputs.close()
        
        return probTb
    }

    fun close() {
        ortSession?.close()
        ortEnv.close()
    }
}
