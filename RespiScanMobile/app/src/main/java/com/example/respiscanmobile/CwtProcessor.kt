package com.example.respiscanmobile

import android.content.Context
import java.io.DataInputStream
import java.io.InputStream
import kotlin.math.abs
import kotlin.math.sqrt

class CwtProcessor(private val context: Context) {

    class WaveletFilter(
        val scale: Float,
        val length: Int,
        val real: FloatArray,
        val imag: FloatArray,
        val startIdx: Int
    )

    private val filters = ArrayList<WaveletFilter>()

    init {
        loadFilters()
    }

    private fun loadFilters() {
        var inputStream: InputStream? = null
        try {
            inputStream = context.assets.open("wavelet_filters.bin")
            val dataInput = DataInputStream(inputStream)
            
            // Read total number of scales
            val numScales = dataInput.readInt()
            for (i in 0 until numScales) {
                val scale = dataInput.readFloat()
                val length = dataInput.readInt()
                
                val real = FloatArray(length)
                for (j in 0 until length) {
                    real[j] = dataInput.readFloat()
                }
                
                val imag = FloatArray(length)
                for (j in 0 until length) {
                    imag[j] = dataInput.readFloat()
                }
                
                // startIdx = floor((L - 2) / 2)
                val startIdx = (length - 2) / 2
                
                filters.add(WaveletFilter(scale, length, real, imag, startIdx))
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            inputStream?.close()
        }
    }

    fun processCwtToTensor(audio: FloatArray): FloatArray {
        val signalLen = audio.size
        // 1. Compute first-order difference: dx[i] = audio[i+1] - audio[i]
        val dx = FloatArray(signalLen - 1)
        for (i in 0 until signalLen - 1) {
            dx[i] = audio[i + 1] - audio[i]
        }
        val dxLen = dx.size

        // 2. Compute CWT matrix of size 64 x 224 (downsampling time dimension in-place)
        val numScales = filters.size
        val targetWidth = 224
        val cwtAbs = Array(numScales) { FloatArray(targetWidth) }
        
        var cMin = Float.MAX_VALUE
        var cMax = Float.MIN_VALUE

        for (s in 0 until numScales) {
            val filter = filters[s]
            val sqrtScale = sqrt(filter.scale)
            
            for (c in 0 until targetWidth) {
                // Map target column to signal index
                val t = (c * (signalLen.toFloat() / targetWidth.toFloat())).toInt().coerceIn(0, signalLen - 1)
                
                // Perform complex convolution at index t with filter h
                var sumReal = 0.0f
                var sumImag = 0.0f
                
                for (m in 0 until filter.length) {
                    val idx = t + filter.startIdx - m
                    if (idx in 0 until dxLen) {
                        val valDx = dx[idx]
                        sumReal += valDx * filter.real[m]
                        sumImag += valDx * filter.imag[m]
                    }
                }
                
                // Multiply by -sqrt(scale)
                val realCoef = -sqrtScale * sumReal
                val imagCoef = -sqrtScale * sumImag
                
                val magnitude = sqrt(realCoef * realCoef + imagCoef * imagCoef)
                cwtAbs[s][c] = magnitude
                
                if (magnitude < cMin) cMin = magnitude
                if (magnitude > cMax) cMax = magnitude
            }
        }

        // 3. Normalize the 64 x 224 CWT matrix to [0, 1]
        val cwtNorm = Array(numScales) { FloatArray(targetWidth) }
        val range = cMax - cMin
        for (s in 0 until numScales) {
            for (c in 0 until targetWidth) {
                cwtNorm[s][c] = if (range > 1e-8f) {
                    (cwtAbs[s][c] - cMin) / range
                } else {
                    0.0f
                }
            }
        }

        // 4. Resize scales from 64 to 224 (Bilinear interpolation in row direction) and apply Jet colormap
        val targetHeight = 224
        val flatTensor = FloatArray(3 * targetHeight * targetWidth)
        
        // ImageNet Normalization Parameters
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        for (r in 0 until targetHeight) {
            // Map target row to scale coordinate
            val y = r * (numScales.toFloat() / targetHeight.toFloat())
            val y0 = y.toInt().coerceIn(0, numScales - 1)
            val y1 = (y0 + 1).coerceIn(0, numScales - 1)
            val wy = y - y0
            
            for (c in 0 until targetWidth) {
                // Bilinear interpolation in the Y (scale) direction
                val v = (1.0f - wy) * cwtNorm[y0][c] + wy * cwtNorm[y1][c]
                
                // Apply Jet Colormap
                val red = clamp(Math.min(4.0f * v - 1.5f, -4.0f * v + 4.5f), 0.0f, 1.0f)
                val green = clamp(Math.min(4.0f * v - 0.5f, -4.0f * v + 3.5f), 0.0f, 1.0f)
                val blue = clamp(Math.min(4.0f * v + 0.5f, -4.0f * v + 2.5f), 0.0f, 1.0f)
                
                // Apply ImageNet normalization: (v - mean) / std
                val normR = (red - mean[0]) / std[0]
                val normG = (green - mean[1]) / std[1]
                val normB = (blue - mean[2]) / std[2]
                
                // Write to NCHW flat array
                flatTensor[0 * targetHeight * targetWidth + r * targetWidth + c] = normR
                flatTensor[1 * targetHeight * targetWidth + r * targetWidth + c] = normG
                flatTensor[2 * targetHeight * targetWidth + r * targetWidth + c] = normB
            }
        }

        return flatTensor
    }

    private fun clamp(value: Float, min: Float, max: Float): Float {
        return Math.max(min, Math.min(value, max))
    }
}
