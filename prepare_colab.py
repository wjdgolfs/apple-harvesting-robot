"""Create a checksum-verified Colab bundle and notebook without changing source data."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parent
DEFAULT_FOLDER = "apple-harvesting-robot-colab-20261116"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_parts(archive_path):
    parts = []
    with archive_path.open("rb") as source:
        index = 1
        while True:
            block = source.read(400 * 1024 * 1024)
            if not block:
                break
            part = archive_path.with_name(archive_path.name + f".{index:03d}")
            part.write_bytes(block)
            parts.append(dict(name=part.name, bytes=len(block),
                              sha256=hashlib.sha256(block).hexdigest()))
            index += 1
    return parts


def notebook(archive_sha, folder, parts):
    cells = []

    def md(source):
        cells.append(dict(cell_type="markdown", metadata={}, source=source.splitlines(keepends=True)))

    def code(source):
        ast.parse(source)
        cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                          source=source.splitlines(keepends=True)))

    md("""# MinneApple YOLOv8n-seg 기본 학습
로컬 검수 완료 데이터: 학습 518장 / 검증 152장 / 사과 주석 28,183개.

1. 상단 **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
2. 아래 셀을 위에서부터 실행합니다. Google Drive 접근 승인 화면은 직접 처리합니다.
3. 데이터는 VM으로 복사해 사용하고, 모델·학습 기록은 매 epoch 후 Drive로 백업합니다.

설정: 640 / batch 1 / mask_ratio 1 / 최대 50 epoch / patience 10 / 증강 없음.
로컬 OOM 시도는 보존하고 COCO 사전학습 가중치부터 새로 학습합니다.
배치·해상도·마스크 비율은 오류 발생 시 자동으로 변경하지 않습니다.
검증 수치는 모델 선택용이며 독립 시험 결과가 아닙니다.

데이터: [MinneApple](https://github.com/nicolaihaeni/MinneApple), Häni, Roy, Isler,
CC BY-NC-SA 3.0 US. 공개용 공유 권한은 변경하지 않습니다.
""")
    code("""import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.215"], check=True)
import torch, ultralytics
subprocess.run(["nvidia-smi"], check=True)
assert torch.cuda.is_available(), "GPU 런타임을 선택하세요."
assert ultralytics.__version__ == "8.4.215"
print("torch:", torch.__version__, "ultralytics:", ultralytics.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
""")
    md("## Google Drive 연결\n인증·접근 승인 화면은 본인이 직접 진행하세요. 이 셀은 승인까지 기다립니다.")
    code("""from google.colab import drive
drive.mount("/content/drive")
""")
    md("## 데이터 복사 및 무결성 확인\nZIP 파일을 Drive에서 VM 디스크로 복사합니다. 같은 분할과 라벨인지 SHA-256 및 객체 수를 검사합니다.")
    code(f"""from pathlib import Path
import hashlib, shutil, zipfile, importlib.util, json
DRIVE_PROJECT = Path("/content/drive/MyDrive") / {folder!r}
PARTS = {parts!r}
LOCAL_ARCHIVE = Path("/content/minneapple_colab_v1.zip")
with LOCAL_ARCHIVE.open("wb") as destination:
    for part in PARTS:
        source = DRIVE_PROJECT / part["name"]
        assert source.is_file(), f"Drive 업로드 확인: {{source}}"
        digest = hashlib.sha256()
        copied = 0
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                destination.write(block)
                digest.update(block)
                copied += len(block)
        assert copied == part["bytes"], f"전송 크기 불일치: {{source.name}}"
        assert digest.hexdigest() == part["sha256"], f"전송 체크섬 불일치: {{source.name}}"
        print("전송 확인:", source.name)
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
assert file_sha256(LOCAL_ARCHIVE) == {archive_sha!r}, "전송 파일 체크섬 불일치"
ROOT = Path("/content/apple_baseline_bundle")
if ROOT.exists():
    raise RuntimeError("이미 압축을 푼 런타임입니다. 검수 결과가 있으면 이 셀을 건너뛰세요.")
with zipfile.ZipFile(LOCAL_ARCHIVE) as archive:
    for member in archive.infolist():
        if not (ROOT / member.filename).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Invalid archive path")
    archive.extractall(ROOT)
spec = importlib.util.spec_from_file_location("apple_colab_runner", ROOT / "train_colab.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.verify_bundle(ROOT)
print("데이터 준비 완료:", ROOT)
""")
    md("""## 기본 학습 실행
완료 전까지 이 셀은 실행 중으로 표시됩니다. 매 epoch의 DRIVE_BACKUP 메시지를 확인하세요.
연결이 끊기더라도 실행 VM이 살아 있을 수 있으므로, 기존 작업 상태를 확인한 뒤 재실행하세요.
오류가 발생하면 마지막 출력과 Drive의 baseline_summary.json을 확인하세요.
""")
    code("""result = runner.main(ROOT, DRIVE_PROJECT / "outputs")
print(json.dumps(result, ensure_ascii=False, indent=2))
""")
    md("## 결과 확인\n학습 성공 여부, 완료 epoch, 저장 경로를 확인합니다. 새 이미지 추론은 다음 단계에서 진행합니다.")
    code("""from IPython.display import display, Image
assert result["status"] == "completed", result
print("완료 epoch:", result["epochs_completed"])
print("Drive 저장:", result["drive_backup"])
run_dir = Path(result["run"])
if (run_dir / "results.png").is_file():
    display(Image(filename=str(run_dir / "results.png")))
display(__import__("pandas").read_csv(run_dir / "results.csv").tail())
""")
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    return dict(cells=cells, metadata=dict(
        accelerator="GPU", colab=dict(name="MinneApple_YOLOv8n_seg_baseline.ipynb", provenance=[]),
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python")), nbformat=4, nbformat_minor=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path(r"C:\26_2\컴퓨터비전\팀플\minneapple_yolo_v1"))
    parser.add_argument("--output", type=Path, default=REPO / "outputs/colab_transfer")
    parser.add_argument("--drive-folder", default=DEFAULT_FOLDER)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "minneapple_colab_v1.zip"
    if archive_path.exists():
        raise FileExistsError(f"Bundle already exists: {archive_path}")
    from ultralytics.utils import YAML
    settings = YAML.load(str(REPO / "configs/train_baseline.yaml"))
    settings.pop("data")
    files = {}
    hashes = {}
    with (REPO / "splits/minneapple_v1/manifest.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts = {"train": {"images": 0, "objects": 0}, "val": {"images": 0, "objects": 0}}
    for row in rows:
        split, name = row["split"], row["filename"]
        if split not in counts:
            raise ValueError(f"Unexpected split: {split}")
        image = dataset / "images" / split / name
        label = dataset / "labels" / split / (Path(name).stem + ".txt")
        image_hash = sha256(image)
        if image_hash != row["image_sha256"]:
            raise ValueError(f"Original image mismatch: {name}")
        lines = [s for s in label.read_text(encoding="utf-8").splitlines() if s.strip()]
        if len(lines) != int(row["instance_id_count"]):
            raise ValueError(f"Label row mismatch: {name}")
        counts[split]["images"] += 1
        counts[split]["objects"] += len(lines)
        for path, relative, digest in (
                (image, f"images/{split}/{name}", image_hash),
                (label, f"labels/{split}/{label.name}", sha256(label))):
            if relative in files:
                raise ValueError(f"Duplicate bundle path: {relative}")
            files[relative] = path
            hashes[relative] = digest
    expected = {"train": {"images": 518, "objects": 22082},
                "val": {"images": 152, "objects": 6101}}
    if counts != expected:
        raise ValueError(counts)
    extras = {
        "train_colab.py": REPO / "colab/train_colab.py",
        "weights/yolov8n-seg.pt": REPO / "weights/yolov8n-seg.pt",
        "splits/manifest.csv": REPO / "splits/minneapple_v1/manifest.csv",
        "splits/summary.json": REPO / "splits/minneapple_v1/summary.json",
        "conversion/summary.json": dataset / "conversion/summary.json",
        "conversion/object_mapping.csv": dataset / "conversion/object_mapping.csv",
    }
    generated = {
        "configs/train_baseline.json": json.dumps(settings, indent=2).encode("utf-8"),
        "DATA_SOURCE.txt": (
            "MinneApple: A Benchmark Dataset for Apple Detection and Segmentation\n"
            "Authors: Nicolai Haeni, Pravakar Roy, Volkan Isler\n"
            "https://github.com/nicolaihaeni/MinneApple\n"
            "Dataset license: CC BY-NC-SA 3.0 United States\n"
            "https://creativecommons.org/licenses/by-nc-sa/3.0/us/\n"
            "Modifications: fixed sequence-wise train/val split; original masks converted to YOLO polygons.\n"
            "Images unchanged. Original source mask IDs mapped in conversion/object_mapping.csv.\n"
            "Original masks and the independent test set are not included in this training bundle.\n"
        ).encode("utf-8"),
    }
    for relative, path in extras.items():
        files[relative] = path
        hashes[relative] = sha256(path)
    for relative, data in generated.items():
        hashes[relative] = hashlib.sha256(data).hexdigest()
    manifest = dict(split=counts, files_sha256=hashes,
                    split_manifest_sha256=hashes["splits/manifest.csv"])
    with zipfile.ZipFile(archive_path, "x", allowZip64=True) as archive:
        for relative, path in files.items():
            compression = zipfile.ZIP_STORED if relative.endswith((".png", ".pt")) else zipfile.ZIP_DEFLATED
            archive.write(path, relative, compress_type=compression)
        for relative, data in generated.items():
            archive.writestr(relative, data, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("bundle_manifest.json", json.dumps(
            manifest, ensure_ascii=False, indent=2), compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC check failed")
        for relative, expected_hash in hashes.items():
            digest = hashlib.sha256()
            with archive.open(relative) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise RuntimeError(f"ZIP member SHA mismatch: {relative}")
    archive_sha = sha256(archive_path)
    parts = write_parts(archive_path)
    notebook_path = REPO / "colab/MinneApple_YOLOv8n_seg_baseline.ipynb"
    notebook_path.write_text(json.dumps(notebook(archive_sha, args.drive_folder, parts),
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    record = dict(status="verified", archive=str(archive_path), archive_bytes=archive_path.stat().st_size,
                  archive_sha256=archive_sha, notebook=str(notebook_path), split=counts,
                  members_checked=len(hashes), parts=parts, drive_folder=args.drive_folder,
                  start_weights="Original COCO weights; local partial checkpoints are preserved, not resumed")
    (output / "transfer_summary.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

