#!/usr/bin/env bash

set -euo pipefail

synapse axon-benchmark-render \
  log-max20b/stream.csv \
  --table-format html > log-max20b/partial.html
