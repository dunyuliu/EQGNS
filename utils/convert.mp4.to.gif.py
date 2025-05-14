#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This script converts all .mp4 files in a given directory and its subdirectories to .gif format.
# It uses ffmpeg for the conversion process.    
import os
import subprocess
import sys

def convert_mp4_to_gif(search_dir):
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".mp4"):
                mp4_file = os.path.join(root, file)
                gif_file = os.path.splitext(mp4_file)[0] + ".gif"
                
                print(f"Converting {mp4_file} to {gif_file}...")
                
                # Use ffmpeg to convert MP4 to GIF
                try:
                    # Remove existing GIF if it exists
                    if os.path.exists(gif_file):
                        print(f"Removing existing file: {gif_file}")
                        os.remove(gif_file)
                    
                    # Convert MP4 to GIF using ffmpeg
                    subprocess.run(
                        [
                            "ffmpeg", "-i", mp4_file, 
                            "-vf", "fps=30,scale=320:-1:flags=lanczos", 
                            "-c:v", "gif", gif_file
                        ],
                        check=True
                    )
                    print(f"Conversion complete: {gif_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {mp4_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory_path>")
        sys.exit(1)
    
    search_dir = sys.argv[1]
    if not os.path.isdir(search_dir):
        print(f"Error: {search_dir} is not a valid directory.")
        sys.exit(1)
    
    convert_mp4_to_gif(search_dir)