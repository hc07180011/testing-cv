#!/bin/sh
pyinstaller --onefile --hiddenimport pywt._extensions._cwt demo.py
pyinstaller --onefile --hiddenimport pywt._extensions._cwt extract_video_encodings.py