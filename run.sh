#!/bin/bash

echo "Running skills_indexing script..."
python skills_indexer.py &

echo "Running job_similarity_indexing script..."
python job_similarity_indexer.py &

wait

echo "Finished all indexing tasks. Exiting..."