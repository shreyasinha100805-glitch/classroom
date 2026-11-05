import cv2
import tensorflow as tf
import easyocr
import sklearn
import spacy
import flask
import importlib.metadata

print("=" * 50)
print("AI CLASSROOM BOARD EXTRACTOR")
print("=" * 50)

print("OpenCV      :", cv2.__version__)
print("TensorFlow  :", tf.__version__)
print("EasyOCR     : OK")
print("Scikit-Learn:", sklearn.__version__)
print("spaCy       :", spacy.__version__)
print("Flask       :", importlib.metadata.version("flask"))

print("=" * 50)
print("ALL LIBRARIES ARE WORKING!")
print("=" * 50)
