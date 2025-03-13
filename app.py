import json
import boto3
import os
import sys

# Use Amazon Titan for generating embeddings

from langchain.embeddings import AmazonTitanEmbeddings