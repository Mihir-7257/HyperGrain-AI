"""
app.py
Interactive Web Dashboard & Chemical Prediction Server for:
"Non-Destructive Prediction and Spatial Visualization of Wheat Grain Protein and Moisture Content"
Serves an interactive web interface displaying model benchmarks, spectral visualizations,
and pixel-by-pixel 3D wheat seed chemical heatmaps.
"""

import os
import sys
import json
import http.server
import socketserver
import webbrowser
import numpy as np
import pandas as pd

PORT = 8501

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable aggressive caching for local dashboard
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_GET(self):
        if self.path == '/api/results':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            csv_path = os.path.join(os.path.dirname(__file__), "outputs", "benchmark_results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                data = df.to_dict(orient='records')
            else:
                data = []
            self.wfile.write(json.dumps(data).encode('utf-8'))
            return
            
        elif self.path == '/api/summary':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            summary = {
                "title": "Grain Quality Prediction Using Hyperspectral Imaging",
                "datasets": {
                    "chemical": "sensAIfood Wheat NIR (150 samples, 950-1650 nm, verified lab Protein & Moisture %)",
                    "spatial": "WheatDual-4V (Mendeley, 1,600 kernels, 300 bands, 4 cultivars: Yangmai 6, Fanmai 8, Xingmai 13, Jinan 17)"
                },
                "models": ["SNV-PLSR", "SNV-SPA-MLR", "1D-CNN", "LSTM", "CLNet"],
                "best_models": {
                    "protein": {"model": "SNV-PLSR", "Rp2": 0.9903, "RMSEP": 0.2524, "RPD": 10.14},
                    "moisture": {"model": "1st-PLSR", "Rp2": 0.9824, "RMSEP": 0.2811, "RPD": 7.55},
                    "deep_learning_moisture": {"model": "CLNet", "Rp2": 0.9474, "RMSEP": 0.4868, "RPD": 4.36}
                }
            }
            self.wfile.write(json.dumps(summary).encode('utf-8'))
            return
            
        return super().do_GET()


def start_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Starting Grain Quality Hyperspectral Dashboard at http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Server running at http://localhost:{PORT} (Press Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server.")


if __name__ == '__main__':
    start_server()
