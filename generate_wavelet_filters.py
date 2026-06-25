import os
import sys
import struct
import numpy as np
import pywt

# Reconfigure stdout to support unicode/emojis in Windows console
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("🚀 Generating pre-computed wavelet filter bank for Android CWT...")
    
    wavelet_name = 'cmor1.5-1.0'
    widths = np.geomspace(2, 186, num=64)
    output_bin = 'wavelet_filters.bin'
    
    # 1. Integrate the Morlet wavelet using PyWavelets
    wavelet = pywt.DiscreteContinuousWavelet(wavelet_name)
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=12)
    int_psi = np.conj(int_psi)  # matching pywt.cwt
    
    # Force single-precision complex float64 -> complex64 for efficiency
    int_psi = np.asarray(int_psi, dtype=np.complex64)
    x = np.asarray(x, dtype=np.float32)
    
    step = x[1] - x[0]
    total_samples = 0
    
    # 2. Write filter coefficients to binary file
    with open(output_bin, 'wb') as f:
        # Write total number of scales (64) as header
        f.write(struct.pack('>I', len(widths)))
        
        for i, scale in enumerate(widths):
            # Evaluate indices matching PyWavelets discrete scaling logic
            j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
            j = j.astype(np.int32)
            
            if j[-1] >= int_psi.size:
                j = np.extract(j < int_psi.size, j)
                
            # Time-reversed scaled wavelet filter (PyWavelets uses [::-1] for conv)
            int_psi_scale = int_psi[j][::-1]
            length = len(int_psi_scale)
            total_samples += length
            
            # Write scale value as float32
            f.write(struct.pack('>f', float(scale)))
            # Write length of filter as int32
            f.write(struct.pack('>I', length))
            
            # Write real coefficients as float32 array
            for val in int_psi_scale.real:
                f.write(struct.pack('>f', float(val)))
                
            # Write imag coefficients as float32 array
            for val in int_psi_scale.imag:
                f.write(struct.pack('>f', float(val)))
                
    print(f"✅ Filter bank generated at: {output_bin}")
    print(f"📊 Stats: 64 scales, total complex filter length: {total_samples} samples (~{total_samples * 8 / 1024:.2f} KB)")
    
    # 3. Verification Read
    with open(output_bin, 'rb') as f:
        num_scales = struct.unpack('>I', f.read(4))[0]
        assert num_scales == 64, f"Mismatch: {num_scales}"
        
        # Test read the first scale
        scale_val = struct.unpack('>f', f.read(4))[0]
        length_val = struct.unpack('>I', f.read(4))[0]
        print(f"🔍 Verification read for first scale: scale={scale_val:.4f}, length={length_val}")
        
if __name__ == '__main__':
    main()
