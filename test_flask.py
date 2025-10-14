#!/usr/bin/env python3
"""
Quick test to identify the Flask route registration issue
"""

from flask import Flask, jsonify, request
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Simple test routes
@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/test-quality', methods=['POST'])
def test_quality():
    return jsonify({
        'success': True,
        'message': 'Quality analysis test endpoint works!'
    })

@app.route('/debug-routes')
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify({'routes': routes})

if __name__ == '__main__':
    print("Testing Flask route registration...")
    print("Routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule} - {list(rule.methods)}")
    
    app.run(host='0.0.0.0', port=5555, debug=False)