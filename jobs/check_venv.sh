#!/usr/bin/env bash
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
.venv/bin/python -c "import numpy, pandas, pyarrow, scipy, sklearn, matplotlib, yaml, tqdm, openai, dotenv, fcsparser, minisom, torch, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('numpy', numpy.__version__, 'pandas', pandas.__version__)"
