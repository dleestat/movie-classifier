mkdir -p data/raw
curl http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz | tar -xzC data/raw --strip-components=1
