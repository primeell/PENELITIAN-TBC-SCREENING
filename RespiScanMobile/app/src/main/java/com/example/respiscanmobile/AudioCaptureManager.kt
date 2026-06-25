package com.example.respiscanmobile

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class AudioCaptureManager {
    companion object {
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val DURATION_SEC = 1.0f
        private const val TOTAL_SAMPLES = (SAMPLE_RATE * DURATION_SEC).toInt() // 16000
    }

    private var audioRecord: AudioRecord? = null
    private var isRecording = false

    @SuppressLint("MissingPermission")
    suspend fun recordCough(): FloatArray = withContext(Dispatchers.IO) {
        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        val shortBuffer = ShortArray(TOTAL_SAMPLES)
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize.coerceAtLeast(TOTAL_SAMPLES * 2)
        )

        val recordedData = FloatArray(TOTAL_SAMPLES)
        var totalSamplesRead = 0

        try {
            audioRecord?.startRecording()
            isRecording = true

            // Read until shortBuffer is full
            while (isRecording && totalSamplesRead < TOTAL_SAMPLES) {
                val readSize = audioRecord?.read(shortBuffer, totalSamplesRead, TOTAL_SAMPLES - totalSamplesRead) ?: -1
                if (readSize > 0) {
                    totalSamplesRead += readSize
                } else if (readSize < 0) {
                    throw RuntimeException("Error reading audio data: status $readSize")
                }
            }
        } finally {
            stopRecording()
        }

        // Convert 16-bit short PCM to normalized float32 [-1.0f, 1.0f]
        for (i in 0 until TOTAL_SAMPLES) {
            recordedData[i] = shortBuffer[i].toFloat() / 32768.0f
        }

        return@withContext recordedData
    }

    fun stopRecording() {
        isRecording = false
        try {
            audioRecord?.let {
                if (it.state == AudioRecord.STATE_INITIALIZED && it.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    it.stop()
                }
                it.release()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        audioRecord = null
    }
}
