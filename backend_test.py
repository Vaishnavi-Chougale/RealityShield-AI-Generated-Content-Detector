#!/usr/bin/env python3
import requests
import sys
import json
import base64
import tempfile
import os
from datetime import datetime

class DeepfakeAPITester:
    def __init__(self, base_url="https://d256623f-58e5-470f-a5b7-1e957f616d9e.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, timeout=30):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, timeout=timeout)
                else:
                    headers['Content-Type'] = 'application/json'
                    response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    resp_data = response.json()
                    print(f"   Response keys: {list(resp_data.keys()) if isinstance(resp_data, dict) else 'Non-dict response'}")
                    return True, resp_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                self.failed_tests.append(f"{name}: Expected {expected_status}, got {response.status_code}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append(f"{name}: {str(e)}")
            return False, {}

    def create_test_image(self):
        """Create a simple test image in base64 format"""
        # Create a simple 100x100 RGB image with some pattern
        from PIL import Image, ImageDraw
        import io
        
        img = Image.new('RGB', (100, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some visual features (not uniform)
        draw.rectangle([10, 10, 50, 50], fill='red')
        draw.ellipse([60, 60, 90, 90], fill='blue')
        draw.line([0, 0, 100, 100], fill='green', width=2)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()

    def create_test_audio(self):
        """Create a simple test audio file"""
        import numpy as np
        import wave
        
        # Generate a simple sine wave
        sample_rate = 44100
        duration = 2  # seconds
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit integers
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            with wave.open(tmp.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            
            os.unlink(tmp.name)
            return audio_bytes

    def test_health(self):
        """Test health endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )
        if success and isinstance(response, dict):
            if response.get('status') == 'ok':
                print("   ✅ Health status is 'ok'")
                return True
            else:
                print(f"   ⚠️ Unexpected health status: {response.get('status')}")
        return success

    def test_analyze_image(self):
        """Test image analysis endpoint"""
        try:
            image_data = self.create_test_image()
            files = {'file': ('test_image.jpg', image_data, 'image/jpeg')}
            
            success, response = self.run_test(
                "Image Analysis",
                "POST",
                "api/analyze/image",
                200,
                files=files,
                timeout=120
            )
            
            if success and isinstance(response, dict):
                required_fields = ['verdict', 'confidence', 'summary', 'scan_id']
                missing_fields = [f for f in required_fields if f not in response]
                if missing_fields:
                    print(f"   ⚠️ Missing required fields: {missing_fields}")
                else:
                    print(f"   ✅ Verdict: {response.get('verdict')}, Confidence: {response.get('confidence')}")
                    return True
            return success
            
        except Exception as e:
            print(f"❌ Image test failed: {str(e)}")
            self.failed_tests.append(f"Image Analysis: {str(e)}")
            return False

    def test_analyze_audio(self):
        """Test audio analysis endpoint"""
        try:
            audio_data = self.create_test_audio()
            files = {'file': ('test_audio.wav', audio_data, 'audio/wav')}
            
            success, response = self.run_test(
                "Audio Analysis",
                "POST",
                "api/analyze/audio",
                200,
                files=files,
                timeout=120
            )
            
            if success and isinstance(response, dict):
                required_fields = ['verdict', 'confidence', 'summary', 'scan_id']
                missing_fields = [f for f in required_fields if f not in response]
                if missing_fields:
                    print(f"   ⚠️ Missing required fields: {missing_fields}")
                else:
                    print(f"   ✅ Verdict: {response.get('verdict')}, Confidence: {response.get('confidence')}")
                    return True
            return success
            
        except Exception as e:
            print(f"❌ Audio test failed: {str(e)}")
            self.failed_tests.append(f"Audio Analysis: {str(e)}")
            return False

    def test_get_scans(self):
        """Test get scans endpoint"""
        success, response = self.run_test(
            "Get Scans",
            "GET",
            "api/scans",
            200
        )
        
        if success and isinstance(response, dict):
            if 'scans' in response and 'total' in response:
                print(f"   ✅ Found {response.get('total')} total scans")
                return True
            else:
                print("   ⚠️ Missing 'scans' or 'total' in response")
        return success

    def test_get_analytics(self):
        """Test analytics endpoint"""
        success, response = self.run_test(
            "Get Analytics",
            "GET",
            "api/analytics",
            200
        )
        
        if success and isinstance(response, dict):
            required_fields = ['total_scans', 'real_count', 'fake_count', 'by_type']
            missing_fields = [f for f in required_fields if f not in response]
            if missing_fields:
                print(f"   ⚠️ Missing required fields: {missing_fields}")
            else:
                print(f"   ✅ Total scans: {response.get('total_scans')}")
                return True
        return success

def main():
    print("🚀 Starting Deepfake Detection API Tests")
    print("=" * 50)
    
    tester = DeepfakeAPITester()
    
    # Run all tests
    tests = [
        tester.test_health,
        tester.test_analyze_image,
        tester.test_analyze_audio,
        tester.test_get_scans,
        tester.test_get_analytics,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {str(e)}")
            tester.failed_tests.append(f"{test.__name__}: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    if tester.failed_tests:
        print("\n❌ Failed Tests:")
        for failure in tester.failed_tests:
            print(f"   - {failure}")
    
    if tester.tests_passed == tester.tests_run:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {len(tester.failed_tests)} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())