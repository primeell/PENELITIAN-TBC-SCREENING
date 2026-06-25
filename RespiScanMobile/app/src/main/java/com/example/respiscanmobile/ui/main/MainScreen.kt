package com.example.respiscanmobile.ui.main

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.navigation3.runtime.NavKey
import com.example.respiscanmobile.AudioCaptureManager
import com.example.respiscanmobile.CwtProcessor
import com.example.respiscanmobile.OnnxClassifier
import kotlinx.coroutines.launch
import kotlin.math.abs

@Composable
fun MainScreen(
    onItemClick: (NavKey) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    // Core Managers (remembered to survive recompositions)
    val audioManager = remember { AudioCaptureManager() }
    val cwtProcessor = remember { CwtProcessor(context) }
    val classifier = remember { OnnxClassifier(context) }

    // Screen States
    var isRecording by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("idle") } // idle, recording, processing, success, error
    var audioData by remember { mutableStateOf<FloatArray?>(null) }
    var probTb by remember { mutableStateOf(0f) }
    var errorMsg by remember { mutableStateOf<String?>(null) }
    var lang by remember { mutableStateOf("id") } // id or en

    // Translations Dictionary
    val textDict = remember {
        mapOf(
            "id" to mapOf(
                "title" to "🩺 RespiScan Mobile",
                "subtitle" to "Penapisan Awal Triage Tuberkulosis Berbasis Akustik Batuk",
                "disclaimer_title" to "⚠️ PENAFIAN MEDIS PENTING:",
                "disclaimer_body" to "RespiScan adalah alat bantu skrining awal (triage) dengan sensitivitas tinggi untuk menyaring suspek TB dari batuk rekaman. Aplikasi ini BUKAN pengganti diagnosis standar emas medis seperti TCM GeneXpert. Hasil penapisan harus divalidasi oleh pemeriksaan klinis dokter.",
                "tap_to_record" to "Ketuk Tombol untuk Merekam Batuk",
                "recording" to "Merekam... Batuk dengan paksa (1 Detik)",
                "processing" to "Memproses CWT & Skrining AI...",
                "triage_result" to "Hasil Penapisan Triage",
                "risk_score" to "Skor Risiko TB",
                "high_risk" to "🚨 RISIKO TINGGI (Suspek TB)",
                "high_risk_desc" to "Karakteristik batuk menyerupai pola tuberkulosis klinis (Probabilitas >= 35.95%). Direkomendasikan melakukan pemeriksaan TCM GeneXpert.",
                "low_risk" to "🟢 RISIKO RENDAH (Non-TB)",
                "low_risk_desc" to "Pola batuk khas tuberkulosis tidak terdeteksi. Hubungi dokter jika gejala pernapasan klinis Anda menetap.",
                "retry" to "Skrining Ulang",
                "grant_perm" to "Izinkan Akses Mikrofon",
                "perm_required" to "Aplikasi memerlukan izin mikrofon untuk merekam suara batuk.",
                "toggle_lang" to "English Version"
            ),
            "en" to mapOf(
                "title" to "🩺 RespiScan Mobile",
                "subtitle" to "Cough Acoustic-Based Tuberculosis Triage Screening Aid",
                "disclaimer_title" to "⚠️ IMPORTANT MEDICAL DISCLAIMER:",
                "disclaimer_body" to "RespiScan is a high-sensitivity triage screening assistant to detect suspected TB from cough recordings. It is NOT a replacement for standard medical diagnostic tests like GeneXpert. All results must be validated by standard clinical diagnostic procedures.",
                "tap_to_record" to "Tap Button to Record Cough",
                "recording" to "Recording... Cough forcefully (1 Second)",
                "processing" to "Processing CWT & AI Inference...",
                "triage_result" to "Triage Screening Outcome",
                "risk_score" to "TB Risk Score",
                "high_risk" to "🚨 HIGH RISK (Suspected TB)",
                "high_risk_desc" to "Cough pattern matches clinical tuberculosis characteristics (Probability >= 35.95%). Standard GeneXpert testing is highly recommended.",
                "low_risk" to "🟢 LOW RISK (Non-TB)",
                "low_risk_desc" to "Cough pattern typical of tuberculosis was not detected. Consult a physician if clinical respiratory symptoms persist.",
                "retry" to "Screen Again",
                "grant_perm" to "Grant Microphone Permission",
                "perm_required" to "The app requires microphone access to record cough audio.",
                "toggle_lang" to "Versi Indonesia"
            )
        )
    }

    val currentText = textDict[lang]!!

    // Microphone Permission Launcher
    var hasMicPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        )
    }
    
    val requestPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasMicPermission = isGranted
        if (!isGranted) {
            status = "error"
            errorMsg = "Izin mikrofon ditolak."
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFFF8FAFC))
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title & Header Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            elevation = CardDefaults.cardElevation(2.dp),
            shape = RoundedCornerShape(0.dp, 0.dp, 16.dp, 16.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = currentText["title"]!!,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF0F172A)
                    )
                    
                    Button(
                        onClick = { lang = if (lang == "id") "en" else "id" },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFE2E8F0), contentColor = Color(0xFF334155)),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(text = currentText["toggle_lang"]!!, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Text(
                    text = currentText["subtitle"]!!,
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFF64748B),
                    textAlign = TextAlign.Center
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Main Interaction Block
        if (!hasMicPermission) {
            // Ask for permission screen
            Card(
                modifier = Modifier.fillMaxWidth(0.9f),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFBEB)),
                border = BorderStroke(1.dp, Color(0xFFF59E0B)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = currentText["grant_perm"]!!,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF92400E)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = currentText["perm_required"]!!,
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color(0xFFB45309),
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Button(
                        onClick = { requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO) },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD97706))
                    ) {
                        Text(text = currentText["grant_perm"]!!)
                    }
                }
            }
        } else {
            // Main control flow
            when (status) {
                "idle", "recording" -> {
                    Text(
                        text = if (isRecording) currentText["recording"]!! else currentText["tap_to_record"]!!,
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium,
                        color = if (isRecording) Color(0xFFEF4444) else Color(0xFF475569)
                    )
                    
                    Spacer(modifier = Modifier.height(32.dp))

                    // Pulsing Mic Record Button
                    Button(
                        onClick = {
                            if (!isRecording) {
                                isRecording = true
                                status = "recording"
                                scope.launch {
                                    try {
                                        val data = audioManager.recordCough()
                                        audioData = data
                                        status = "processing"
                                        
                                        // Run Feature Processing & Inference in background
                                        val flatTensor = cwtProcessor.processCwtToTensor(data)
                                        val prob = classifier.classifyCough(flatTensor)
                                        probTb = prob
                                        status = "success"
                                    } catch (e: Exception) {
                                        e.printStackTrace()
                                        status = "error"
                                        errorMsg = e.message ?: "Unknown recording error."
                                    } finally {
                                        isRecording = false
                                    }
                                }
                            }
                        },
                        enabled = !isRecording,
                        modifier = Modifier.size(120.dp),
                        shape = CircleShape,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isRecording) Color(0xFFEF4444) else Color(0xFF0284C7),
                            disabledContainerColor = Color(0xFFEF4444)
                        ),
                        elevation = ButtonDefaults.buttonElevation(defaultElevation = 6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Mic,
                            contentDescription = "Record",
                            modifier = Modifier.size(48.dp),
                            tint = Color.White
                        )
                    }
                }
                
                "processing" -> {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = Color(0xFF0284C7), strokeWidth = 4.dp)
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(text = currentText["processing"]!!, color = Color(0xFF0284C7), fontWeight = FontWeight.Medium)
                    }
                }
                
                "success" -> {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth(0.9f)
                    ) {
                        // Visual Waveform Canvas
                        audioData?.let { audio ->
                            Text(
                                text = if (lang == "id") "Sinyal Akustik Batuk Terpilih" else "Selected Cough Acoustic Signal",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFF475569)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(90.dp)
                                    .background(Color.White, RoundedCornerShape(8.dp))
                                    .border(1.dp, Color(0xFFE2E8F0), RoundedCornerShape(8.dp))
                                    .padding(8.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Canvas(modifier = Modifier.fillMaxSize()) {
                                    val width = size.width
                                    val height = size.height
                                    val centerY = height / 2f
                                    val step = (audio.size / 150).coerceAtLeast(1)
                                    val pointsCount = audio.size / step
                                    val barWidth = width / pointsCount.toFloat()
                                    
                                    for (i in 0 until pointsCount) {
                                        val sampleIndex = i * step
                                        if (sampleIndex < audio.size) {
                                            val value = audio[sampleIndex]
                                            val barHeight = (abs(value) * height * 0.9f).coerceAtMost(height)
                                            val x = i * barWidth
                                            drawLine(
                                                color = Color(0xFF0284C7),
                                                start = Offset(x, centerY - barHeight / 2f),
                                                end = Offset(x, centerY + barHeight / 2f),
                                                strokeWidth = 2.5f
                                            )
                                        }
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(24.dp))

                        // Triage Status Card
                        val isHighRisk = probTb >= OnnxClassifier.DECISION_THRESHOLD
                        val bannerColor = if (isHighRisk) Color(0xFFFEE2E2) else Color(0xFFECFDF5)
                        val borderColor = if (isHighRisk) Color(0xFFFCA5A5) else Color(0xFFA7F3D0)
                        
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = bannerColor),
                            border = BorderStroke(1.dp, borderColor)
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Text(
                                    text = if (isHighRisk) currentText["high_risk"]!! else currentText["low_risk"]!!,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 16.sp,
                                    color = if (isHighRisk) Color(0xFF991B1B) else Color(0xFF065F46)
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = if (isHighRisk) currentText["high_risk_desc"]!! else currentText["low_risk_desc"]!!,
                                    fontSize = 13.sp,
                                    color = if (isHighRisk) Color(0xFF7F1D1D) else Color(0xFF064E3B)
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(24.dp))

                        // Custom Circular Risk Gauge
                        Box(
                            contentAlignment = Alignment.Center,
                            modifier = Modifier.size(150.dp)
                        ) {
                            CircularProgressIndicator(
                                progress = probTb,
                                modifier = Modifier.fillMaxSize(),
                                strokeWidth = 12.dp,
                                color = if (isHighRisk) Color(0xFFEF4444) else Color(0xFF10B981),
                                trackColor = Color(0xFFE2E8F0)
                            )
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(
                                    text = String.format("%.1f%%", probTb * 100f),
                                    fontSize = 28.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFF0F172A)
                                )
                                Text(
                                    text = currentText["risk_score"]!!,
                                    fontSize = 10.sp,
                                    color = Color(0xFF64748B),
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(24.dp))

                        // Reset button to classify again
                        Button(
                            onClick = { status = "idle" },
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF475569))
                        ) {
                            Icon(imageVector = Icons.Default.Refresh, contentDescription = "Retry")
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(text = currentText["retry"]!!)
                        }
                    }
                }
                
                "error" -> {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth(0.9f)
                    ) {
                        Text(text = "❌ Error", color = Color.Red, fontWeight = FontWeight.Bold)
                        Text(text = errorMsg ?: "Unknown error", color = Color.DarkGray)
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(onClick = { status = "idle" }) {
                            Text(text = currentText["retry"]!!)
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Standard medical disclaimer box at bottom
        Card(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .padding(bottom = 32.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFEF2)),
            border = BorderStroke(1.dp, Color(0xFFFEF08A))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = currentText["disclaimer_title"]!!,
                    fontWeight = FontWeight.Bold,
                    fontSize = 11.sp,
                    color = Color(0xFF854D0E)
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = currentText["disclaimer_body"]!!,
                    fontSize = 10.sp,
                    lineHeight = 14.sp,
                    color = Color(0xFF713F12),
                    textAlign = TextAlign.Justify
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {
    MainScreen(onItemClick = {})
}
