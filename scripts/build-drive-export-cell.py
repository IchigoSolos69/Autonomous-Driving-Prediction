"""Generate a foreground export-only cell without altering the pinned training bundle."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "artifacts/cloud-queue-v1/colab-queue-save-to-drive.py"
TAIL = '''
# Explicit user-approved mount. This invokes Google's normal account permission flow.
from google.colab import drive
drive.mount('/content/drive')
if not os.path.ismount('/content/drive'):
    raise RuntimeError('Google Drive is not mounted; no local substitute will be treated as durable storage.')
ROOT = Path('/content/interaction-lab-colab-queue-v1')
DESTINATION = Path('/content/drive/MyDrive/InteractionLab/compute-v1/colab/development-v1')
if not (ROOT/'worker-status.json').is_file():
    raise RuntimeError('The previous runtime queue is unavailable. Do not recreate completed jobs or reset their budget.')
# Export existing results only: no worker is started, restarted or reset.
saved = publish_saved_queue(ROOT, DESTINATION, 'colab')
print(json.dumps({'saved_jobs':saved['completed_jobs'], 'saved_epochs':saved['completed_epochs'],
    'exports':len(saved['exports']), 'destination':str(DESTINATION),
    'notice':'Checksummed saved evidence; strict local CPU verification is still required.'}, sort_keys=True))
'''


def main():
    source = (PROJECT / "src/adp/drive_evidence.py").read_text(encoding="utf-8") + "\n" + TAIL
    compile(source, OUTPUT.name, "exec")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(source, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
