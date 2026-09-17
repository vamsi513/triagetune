# Data provenance and reuse

## Banking intent data

The tracked source files under `data/raw/` are copies of the public BANKING77 training and test files from the [dataset publisher](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data). The publisher licenses the dataset under [Creative Commons Attribution 4.0 International](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE). Please retain this attribution when reusing the data. The dataset is described by Casanueva et al., “Efficient Intent Detection with Dual Sentence Encoders” (2020).

| Tracked source file | SHA-256 |
| --- | --- |
| `data/raw/train.csv` | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| `data/raw/test.csv` | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

The files under `data/processed/` are project-created cleaned and split derivatives of those source files. The published test records remain in the test split. Preparation details and duplicate findings are recorded in `reports/data_preparation_report.json`. Prediction records and reports may include original query text and should be handled under the dataset's license as well.

## Other evaluation text

`data/robustness_probes.jsonl` and `data/synthetic/unknown_request_probes.jsonl` are project-created synthetic examples, not customer traffic. Every report using them labels the results as synthetic.

The external intent dataset used for one scope challenge is from [Larson et al. (2019)](https://github.com/clinc/oos-eval) and is licensed under [Creative Commons Attribution 3.0 Unported](https://github.com/clinc/oos-eval/blob/master/LICENSE). Its raw file is cached locally and is not redistributed here. The exact source revision, file checksum, selection rules, and limitations are recorded in `reports/external_scope_evaluation.json`.

## Code and model files

The saved adapter, base-model cache, and classical-model artifact are ignored local files and are not distributed in this repository. The repository owner's project-authored source code and documentation are licensed under the root Apache 2.0 `LICENSE`. That license does not relicense the BANKING77 source files, their processed derivatives, or benchmark query text reproduced in prediction reports; those materials remain subject to the publisher's CC BY 4.0 license and attribution. The separately published adapter has its own Apache 2.0 license.
