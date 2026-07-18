# Log Evaluation Project

This project evaluates anomaly-detection approaches for log data.

Focus areas:

- log parsing and normalization
- sequence and session feature extraction
- anomaly scoring and ranking
- structured result export

# CLIs
python -m src.embed_similarity --input ./data/test.csv --text-column log --output ./data/output.csv

python -m src.log_template --input ./data/log-raw.csv --text-column message --output ./data/template_output.csv
