"""Build the private RL payload locally; upload only with explicit --upload."""
import argparse
import ast
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT_REPO = 'GL3MON/tau2-retail-rl-job-input'
RESULT_REPO = 'GL3MON/tau2-retail-rl-job-results'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload', action='store_true', help='Explicitly upload to private HF datasets; does not launch compute')
    parser.add_argument('--tau2-dir', type=Path, default=ROOT.parent / 'tau2-bench')
    args = parser.parse_args()
    artifact_dir = HERE / 'artifacts'
    artifact_dir.mkdir(exist_ok=True)
    archive_path = artifact_dir / 'payload.tar.gz'
    adapter = ROOT / 'outputs/rl-initialization/retail-sdg-v2-checkpoint-100'
    if json.loads((adapter / 'trainer_state.json').read_text())['global_step'] != 100:
        raise ValueError('Expected verified checkpoint 100')
    expected_adapter_sha256 = '97bf4daa0de4fcc48a3ae31c6cd60f77d916fcb4d6ada11cff8191fbcfc9e176'
    if hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest() != expected_adapter_sha256:
        raise ValueError('Checkpoint 100 adapter differs from the verified initialization')
    if (adapter/'chat_template.jinja').read_text() != (ROOT/'templates/retail_sdg_v2.jinja').read_text():
        raise ValueError('RL template differs from checkpoint 100 SFT template')
    added = []
    with tarfile.open(archive_path, 'w:gz', compresslevel=3) as archive:
        def add_file(source, destination):
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.add(source, arcname=destination, recursive=False)
            added.append(destination)

        def add_tree(source, destination):
            for file in sorted(source.rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts and file.suffix not in {'.pyc', '.wav', '.mp3', '.flac'}:
                    add_file(file, str(Path(destination) / file.relative_to(source)))

        add_tree(ROOT / 'src', 'project/src')
        add_tree(ROOT / 'templates', 'project/templates')
        for filename in ['pyproject.toml', 'README.md']:
            add_file(ROOT / filename, 'project/' + filename)
        add_file(HERE / 'smoke-config.json', 'project/hf-smoke-config.json')
        add_file(ROOT / 'outputs/retail-data/split_manifest.json', 'project/split_manifest.json')
        for filename in ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json',
                         'tokenizer.json', 'chat_template.jinja', 'trainer_state.json', 'provenance.json']:
            add_file(adapter / filename, 'project/initial-adapter/' + filename)
        add_tree(args.tau2_dir / 'src', 'tau2-bench/src')
        for filename in ['pyproject.toml', 'README.md', 'LICENSE']:
            add_file(args.tau2_dir / filename, 'tau2-bench/' + filename)
        data = args.tau2_dir / 'data/tau2'
        for relative in ['domains/retail', 'multilingual', 'multilingual_extensions',
                         'judge_catalog', 'benchmarks', 'task_subsets', 'voice_design']:
            add_tree(data / relative, 'tau2-bench/data/tau2/' + relative)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {'input_repo': INPUT_REPO, 'result_repo': RESULT_REPO,
                'payload_sha256': digest, 'payload_bytes': archive_path.stat().st_size,
                'files': added, 'initial_checkpoint': 100,
                'base_model': 'GL3MON/tiny-aya-global-function-calling-sft',
                'base_revision': 'df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a',
                'contains_credentials': False, 'compute_launched': False}
    (artifact_dir / 'payload-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for repo in [INPUT_REPO, RESULT_REPO]:
            api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
            if not api.repo_info(repo, repo_type='dataset').private:
                raise RuntimeError('Refusing to use a public payload/results repository')
        for local, remote in [(archive_path, 'payload.tar.gz'),
                              (artifact_dir / 'payload-manifest.json', 'payload-manifest.json'),
                              (HERE / 'bootstrap.py', 'bootstrap.py')]:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=INPUT_REPO,
                            repo_type='dataset', commit_message='Prepare private Tau2 RL smoke inputs')
        print('Private payload uploaded. No compute job submitted.')


if __name__ == '__main__':
    main()
