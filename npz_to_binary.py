#!/usr/bin/env python3
"""
NPZ to Binary JSON Converter

This script converts numpy NPZ files to a more efficient binary JSON format (BSON)
for faster loading in web applications. It preserves all the original metadata.

Requirements:
- numpy
- bson (pip install bson)
- os, glob, json, datetime (standard library)

Usage:
1. Place this script in the same directory as your numerical_data folder
2. Run: python npz_to_bson_converter.py
3. It will create a 'converted_data' folder with all converted files
"""

import os
import glob
import json
import numpy as np
import bson
from datetime import datetime


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


def convert_npz_to_bson(npz_filepath, output_dir):
    """Convert a numpy NPZ file to BSON format."""
    filename = os.path.basename(npz_filepath)
    base_filename = os.path.splitext(filename)[0]
    output_filepath = os.path.join(output_dir, f"{base_filename}.bson")

    try:
        # Load the NPZ file
        data = np.load(npz_filepath, allow_pickle=True)

        # Create a dictionary to hold all data
        converted_data = {
            'filename': filename,
            'conversion_timestamp': str(datetime.now())
        }

        # Extract all arrays and metadata
        for key in data.files:
            value = data[key]
            converted_data[key] = convert_numpy_types(value)

        # Save as BSON (binary JSON) for faster loading
        with open(output_filepath, 'wb') as f:
            f.write(bson.dumps(converted_data))

        print(f"Converted: {filename} → {os.path.basename(output_filepath)}")
        return True

    except Exception as e:
        print(f"Error converting {filename}: {str(e)}")
        return False


def main():
    # Define directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'numerical_data')
    output_dir = os.path.join(script_dir, 'converted_data')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all NPZ files
    npz_files = glob.glob(os.path.join(input_dir, '*.npz'))

    if not npz_files:
        print(f"No NPZ files found in {input_dir}")
        return

    print(f"Found {len(npz_files)} NPZ files to convert")

    # Convert each file
    success_count = 0
    for npz_file in npz_files:
        if convert_npz_to_bson(npz_file, output_dir):
            success_count += 1

    print(f"Conversion complete: {success_count}/{len(npz_files)} files converted successfully")
    print(f"Converted files saved to: {output_dir}")

    # Create a tiny HTML file to load the converted files
    index_html = """<!DOCTYPE html>
<html>
<head>
    <title>Matrix Viewer (with BSON format)</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        select { padding: 5px; margin: 10px 0; }
        button { padding: 5px 10px; margin-right: 10px; }
        #matrixInfo { margin-top: 20px; white-space: pre; font-family: monospace; }
    </style>
</head>
<body>
    <h1>Fast Matrix Viewer</h1>
    <div>
        <select id="fileSelector">
            <!-- Will be populated with available BSON files -->
        </select>
        <button id="loadBtn">Load Matrix</button>
    </div>
    <div id="matrixInfo"></div>

    <script>
        // Load available BSON files
        async function loadAvailableFiles() {
            try {
                const response = await fetch('/list_converted_files');
                const files = await response.json();

                const selector = document.getElementById('fileSelector');
                selector.innerHTML = '';

                files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file;
                    option.textContent = file;
                    selector.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading file list:', error);
            }
        }

        // Load matrix from BSON file
        async function loadMatrix() {
            const filename = document.getElementById('fileSelector').value;
            const infoDiv = document.getElementById('matrixInfo');

            if (!filename) {
                infoDiv.textContent = 'Please select a file';
                return;
            }

            try {
                infoDiv.textContent = 'Loading...';

                const response = await fetch(`/converted_data/${filename}`);
                const arrayBuffer = await response.arrayBuffer();

                // Use BSON.js library to decode
                const data = BSON.deserialize(new Uint8Array(arrayBuffer));

                // Display matrix info
                let info = `Filename: ${data.filename}\n`;
                info += `B-Field: ${data.B_tesla} T\n`;
                info += `Timestamp: ${data.timestamp}\n`;
                info += `Matrix size: ${data.H.length}x${data.H[0].length}\n`;
                info += `\nSample (first 5x5):\n`;

                // Display a small sample of the matrix
                for (let i = 0; i < Math.min(5, data.H.length); i++) {
                    let row = '';
                    for (let j = 0; j < Math.min(5, data.H[i].length); j++) {
                        row += data.H[i][j].toFixed(4).padStart(10) + ' ';
                    }
                    info += row + '\n';
                }

                infoDiv.textContent = info;
            } catch (error) {
                infoDiv.textContent = `Error loading matrix: ${error.message}`;
                console.error('Error:', error);
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadAvailableFiles();
            document.getElementById('loadBtn').addEventListener('click', loadMatrix);
        });
    </script>

    <!-- Add BSON.js for handling binary JSON -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bson/4.7.2/bson.bundle.min.js"></script>
</body>
</html>
    """

    with open(os.path.join(script_dir, 'fast_viewer.html'), 'w') as f:
        f.write(index_html)

    print(f"Created fast_viewer.html for testing the converted files")


if __name__ == "__main__":
    main()