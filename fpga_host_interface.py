"""
FPGA Host Interface Script
Sends test data to FPGA and verifies output

Features:
1. Load test vectors from pickle file
2. Send test samples to FPGA via UART
3. Receive predictions and confidence scores
4. Compare with Python model predictions
5. Generate accuracy reports
"""

import serial
import numpy as np
import pickle
import json
import time
from pathlib import Path
import argparse

class FPGAInterface:
    """Interface for communicating with FPGA via UART"""
    
    def __init__(self, port='COM3', baudrate=115200, timeout=2.0):
        """
        Initialize UART connection to FPGA
        
        Args:
            port: Serial port (e.g., 'COM3', '/dev/ttyUSB0')
            baudrate: UART baud rate (default 115200)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.test_vectors = None
        self.results = {
            'fpga_predictions': [],
            'python_predictions': [],
            'confidence_scores': [],
            'latencies': []
        }
    
    def connect(self):
        """Establish UART connection"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            print(f"✅ Connected to {self.port} @ {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close UART connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("✅ Disconnected from FPGA")
    
    def send_data(self, data: bytes):
        """Send data to FPGA"""
        if not self.ser or not self.ser.is_open:
            print("❌ Serial port not open")
            return False
        
        try:
            self.ser.write(data)
            return True
        except serial.SerialException as e:
            print(f"❌ Send failed: {e}")
            return False
    
    def receive_data(self, num_bytes: int) -> bytes:
        """Receive data from FPGA"""
        if not self.ser or not self.ser.is_open:
            print("❌ Serial port not open")
            return None
        
        try:
            data = self.ser.read(num_bytes)
            if len(data) < num_bytes:
                print(f"⚠️  Received {len(data)}/{num_bytes} bytes")
            return data
        except serial.SerialException as e:
            print(f"❌ Receive failed: {e}")
            return None
    
    def load_test_vectors(self, pickle_file: str):
        """Load test vectors from pickle file"""
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            self.test_vectors = data
            print(f"✅ Loaded test vectors from {pickle_file}")
            print(f"   Samples: {len(data['ground_truth'])}")
            print(f"   FP32 Accuracy (Python): {data['accuracy_fp32']:.2f}%")
            return True
        except Exception as e:
            print(f"❌ Failed to load test vectors: {e}")
            return False
    
    def send_test_sample(self, sample_idx: int) -> bool:
        """Send a single test sample to FPGA"""
        if not self.test_vectors:
            print("❌ No test vectors loaded")
            return False
        
        if sample_idx >= len(self.test_vectors['ground_truth']):
            print(f"❌ Sample index out of range: {sample_idx}")
            return False
        
        # Get sample data
        sample = self.test_vectors['test_vectors_int8'][sample_idx]  # (4, 3, 64, 64)
        
        # Flatten to bytes
        sample_flat = sample.flatten().astype(np.uint8)
        sample_bytes = sample_flat.tobytes()
        
        print(f"\n📤 Sending sample {sample_idx}...")
        print(f"   Size: {len(sample_bytes)} bytes")
        
        # Send header (sample index + length)
        header = np.array([sample_idx, len(sample_bytes) // 256], dtype=np.uint8).tobytes()
        self.send_data(header)
        
        # Send data in chunks (256 bytes at a time)
        chunk_size = 256
        for i in range(0, len(sample_bytes), chunk_size):
            chunk = sample_bytes[i:i+chunk_size]
            self.send_data(chunk)
            time.sleep(0.01)  # Small delay between chunks
        
        return True
    
    def receive_prediction(self) -> tuple:
        """Receive prediction result from FPGA"""
        # Receive: [predicted_class (1 byte), confidence (1 byte), padding (6 bytes)]
        result = self.receive_data(8)
        
        if not result or len(result) < 2:
            print("❌ Failed to receive prediction")
            return None, None
        
        predicted_class = result[0]
        confidence = result[1]
        
        return predicted_class, confidence
    
    def run_inference_test(self, sample_idx: int, verbose=True) -> bool:
        """Run a single inference test"""
        start_time = time.time()
        
        # Send test sample
        if not self.send_test_sample(sample_idx):
            return False
        
        # Wait for FPGA to process
        print("   ⏳ Waiting for FPGA inference...")
        time.sleep(1.0)  # Adjust based on FPGA latency
        
        # Receive prediction
        pred_class, confidence = self.receive_prediction()
        
        if pred_class is None:
            return False
        
        elapsed = time.time() - start_time
        
        # Get ground truth and Python prediction
        gt = self.test_vectors['ground_truth'][sample_idx]
        python_pred = self.test_vectors['predictions_fp32'][sample_idx]['pred']
        python_conf = int(self.test_vectors['predictions_fp32'][sample_idx]['confidence'] * 255)
        
        # Check accuracy
        fpga_correct = (pred_class == gt)
        python_correct = (python_pred == gt)
        
        if verbose:
            status_fpga = "✅" if fpga_correct else "❌"
            status_python = "✅" if python_correct else "❌"
            
            print(f"\n📊 Sample {sample_idx} Results:")
            print(f"   Ground Truth: {gt}")
            print(f"   FPGA Prediction: {pred_class} {status_fpga} (confidence={confidence}/255)")
            print(f"   Python Prediction: {python_pred} {status_python} (confidence={python_conf}/255)")
            print(f"   Latency: {elapsed:.3f}s")
        
        # Store results
        self.results['fpga_predictions'].append(pred_class)
        self.results['python_predictions'].append(python_pred)
        self.results['confidence_scores'].append(confidence)
        self.results['latencies'].append(elapsed)
        
        return fpga_correct
    
    def run_batch_test(self, num_samples=None, start_sample=0):
        """Run inference on multiple test samples"""
        if not self.test_vectors:
            print("❌ No test vectors loaded")
            return
        
        num_samples = num_samples or len(self.test_vectors['ground_truth'])
        num_samples = min(num_samples, len(self.test_vectors['ground_truth']))
        
        print("="*80)
        print("BATCH INFERENCE TEST")
        print("="*80)
        print(f"Testing {num_samples} samples starting from {start_sample}")
        
        correct_count = 0
        
        for i in range(start_sample, start_sample + num_samples):
            if self.run_inference_test(i, verbose=True):
                correct_count += 1
        
        # Print summary
        accuracy = 100 * correct_count / num_samples
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Samples tested: {num_samples}")
        print(f"Correct predictions: {correct_count}/{num_samples}")
        print(f"FPGA Accuracy: {accuracy:.2f}%")
        print(f"Average Latency: {np.mean(self.results['latencies']):.3f}s")
        
        return accuracy
    
    def compare_with_python(self):
        """Compare FPGA predictions with Python model predictions"""
        if not self.results['fpga_predictions']:
            print("❌ No results to compare")
            return
        
        fpga_preds = np.array(self.results['fpga_predictions'])
        python_preds = np.array(self.results['python_predictions'])
        gts = np.array(self.test_vectors['ground_truth'][:len(fpga_preds)])
        
        fpga_acc = 100 * np.mean(fpga_preds == gts)
        python_acc = 100 * np.mean(python_preds == gts)
        agreement = 100 * np.mean(fpga_preds == python_preds)
        
        print("\n" + "="*80)
        print("FPGA vs PYTHON COMPARISON")
        print("="*80)
        print(f"FPGA Accuracy: {fpga_acc:.2f}%")
        print(f"Python Accuracy: {python_acc:.2f}%")
        print(f"Agreement: {agreement:.2f}%")
        
        if agreement < 100:
            print("\n⚠️  Disagreements detected!")
            mismatches = np.where(fpga_preds != python_preds)[0]
            for idx in mismatches[:5]:  # Show first 5 mismatches
                print(f"   Sample {idx}: FPGA={fpga_preds[idx]}, Python={python_preds[idx]}, GT={gts[idx]}")


def main():
    parser = argparse.ArgumentParser(description="FPGA Host Interface")
    parser.add_argument('--port', default='COM3', help='Serial port (default: COM3)')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--test-vectors', default='./fpga_test_vectors/fpga_test_vectors.pkl',
                        help='Path to test vectors pickle file')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to test (default: 5)')
    parser.add_argument('--start-sample', type=int, default=0,
                        help='Starting sample index (default: 0)')
    parser.add_argument('--offline', action='store_true',
                        help='Run offline verification (no FPGA connection)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TINYMOBILENET FPGA VERIFICATION TOOL")
    print("="*80)
    
    # Initialize interface
    interface = FPGAInterface(port=args.port, baudrate=args.baud)
    
    # Load test vectors
    if not interface.load_test_vectors(args.test_vectors):
        return
    
    if args.offline:
        print("\n🔵 Running in OFFLINE mode (no FPGA connection)")
        print("   This will verify the test data only")
        interface.compare_with_python()
    else:
        # Connect to FPGA
        print(f"\n🔵 Connecting to FPGA on {args.port}...")
        if not interface.connect():
            print("❌ Could not connect to FPGA. Use --offline flag to run without FPGA.")
            return
        
        try:
            # Run batch test
            interface.run_batch_test(
                num_samples=args.num_samples,
                start_sample=args.start_sample
            )
            
            # Compare with Python
            interface.compare_with_python()
        finally:
            interface.disconnect()


if __name__ == "__main__":
    main()
