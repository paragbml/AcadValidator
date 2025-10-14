#!/usr/bin/env python3
"""
Test script to verify AI insights endpoint is working
"""
import requests
import json

def test_ai_insights():
    """Test the AI insights endpoint"""
    url = "http://localhost:5001/api/ai-insights"
    
    test_data = {
        "query": "What are the main factors affecting institutional performance?",
        "context": "quality_analysis"
    }
    
    try:
        print("🧪 Testing AI Insights Endpoint...")
        print(f"📡 POST {url}")
        print(f"📤 Data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📥 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ AI Insights Endpoint Working!")
            print(f"📝 AI Response Preview: {result.get('ai_response', 'No response')[:200]}...")
            return True
        else:
            print(f"❌ AI Insights Endpoint Failed!")
            print(f"📝 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    success = test_ai_insights()
    print(f"\n🎯 Test Result: {'PASSED' if success else 'FAILED'}")