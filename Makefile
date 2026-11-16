.PHONY: setup doctor smoke train small test sft sft-prepare sft-split sft-smoke
setup:
	bash scripts/setup.sh
doctor:
	.venv/bin/tau2-grpo doctor
smoke:
	.venv/bin/tau2-grpo smoke
train:
	.venv/bin/tau2-grpo train
small:
	.venv/bin/tau2-grpo train --config configs/small.json
test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v
sft:
	bash scripts/run_sft.sh
sft-prepare:
	bash scripts/run_sft.sh --prepare-only
sft-split:
	bash scripts/prepare_sft_data.sh
sft-smoke:
	bash scripts/smoke_sft.sh
