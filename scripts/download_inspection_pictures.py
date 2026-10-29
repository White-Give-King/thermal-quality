import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
CHUNK_SIZE = 1024 * 1024  # 1 MB


def safe_filename_part(value: object) -> str:
    text = INVALID_FILENAME_CHARS.sub("_", str(value or "").strip())
    return text.rstrip(" .") or "未命名"


def load_records(json_path: Path) -> tuple[str, list[dict]]:
    with json_path.open("r", encoding="utf-8-sig") as file:
        document = json.load(file)

    data = document.get("data")
    if not isinstance(data, dict):
        raise ValueError("JSON 中没有有效的 data 对象")

    dept_name = str(data.get("deptName") or "")
    records = []
    types = data.get("inspectionDeviceTypes", [])
    if not isinstance(types, list):
        raise ValueError("data.inspectionDeviceTypes 不是数组")

    for device_type in types:
        if not isinstance(device_type, dict):
            continue
        devices = device_type.get("inspectionDevices", [])
        if not isinstance(devices, list):
            continue
        records.extend(device for device in devices if isinstance(device, dict))
    return dept_name, records


def _head_file_size(url: str) -> int | None:
    """尝试用 HEAD 请求获取文件大小，失败则返回 None。"""
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "Python inspection picture downloader"})
        with urlopen(req, timeout=15) as resp:
            length = resp.headers.get("Content-Length")
            if length is not None:
                return int(length)
    except (HTTPError, URLError, OSError, ValueError):
        pass
    return None


def download_picture(url: str, target: Path, retries: int = 5) -> None:
    """
    下载单张图片，支持断点续传。
    - 如果 target.jpg 已存在且非空，则跳过。
    - 如果 target.jpg.part 存在且大小 < 远端文件大小，则从该位置继续。
    """
    temporary_target = target.with_name(f"{target.name}.part")

    # 已下载完成，直接跳过
    if target.exists() and target.stat().st_size > 0:
        return

    # 清理 0 字节残留
    if temporary_target.exists() and temporary_target.stat().st_size == 0:
        temporary_target.unlink()

    # 先探测远端文件总大小（用于判断是否应该续传）
    remote_size = _head_file_size(url)

    start_byte = 0
    if temporary_target.exists():
        local_size = temporary_target.stat().st_size
        if remote_size is not None and local_size >= remote_size:
            # 本地已经完整了，直接重命名
            temporary_target.replace(target)
            return
        if remote_size is None or local_size < remote_size:
            start_byte = local_size

    attempt = 0
    while True:
        attempt += 1
        try:
            headers = {"User-Agent": "Python inspection picture downloader"}
            if start_byte > 0:
                headers["Range"] = f"bytes={start_byte}-"

            request = Request(url, headers=headers)
            with urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())

                # 服务器不支持 Range，从头下载
                if start_byte > 0 and status != 206:
                    start_byte = 0
                    headers.pop("Range", None)
                    request = Request(url, headers=headers)
                    response = urlopen(request, timeout=60)

                # 再试一次探测远端总长度（有些服务 GET 时才给）
                if remote_size is None:
                    cl = response.headers.get("Content-Length")
                    if cl is not None:
                        cl_int = int(cl)
                        # 206 时 Content-Length 是剩余部分大小
                        remote_size = start_byte + cl_int if status == 206 else cl_int

                mode = "ab" if (start_byte > 0 and status == 206) else "wb"
                with temporary_target.open(mode) as file:
                    if mode == "wb" and start_byte > 0:
                        # 不应该走到这里，因为我们已在上面重置了 start_byte
                        pass
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        file.write(chunk)

            # 再次校验完整性
            if remote_size is not None:
                actual = temporary_target.stat().st_size
                if actual < remote_size:
                    # 不完整，继续续传
                    start_byte = actual
                    continue

            temporary_target.replace(target)
            return

        except (HTTPError, URLError, OSError, ValueError) as error:
            if attempt >= retries:
                raise
            delay = min(5 * 2 ** min(attempt - 1, 4), 60)
            # 更新续传起点
            if temporary_target.exists():
                start_byte = temporary_target.stat().st_size
            print(
                f"    连接中断（{error}），已下载 {start_byte} 字节；"
                f"第 {attempt}/{retries} 次重试将在 {delay} 秒后从断点继续……",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def collect_tasks(json_files: list[Path], output_root: Path | None) -> list[tuple[Path, str, Path]]:
    """返回 (json_path, picture_url, target_path) 列表。"""
    tasks: list[tuple[Path, str, Path]] = []
    for json_file in json_files:
        try:
            dept_name, records = load_records(json_file)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"[跳过] {json_file.name}: {error}", file=sys.stderr)
            continue

        if output_root is not None:
            # 单个 JSON 时 -o 即最终目录；多个 JSON 时按 JSON 名各建子目录
            output_dir = (
                output_root
                if len(json_files) == 1
                else output_root / json_file.stem
            )
        else:
            output_dir = json_file.parent / json_file.stem

        for device in records:
            picture_url = device.get("picture")
            if not isinstance(picture_url, str) or not picture_url.strip():
                continue
            parts = (
                dept_name,
                device.get("name"),
                dept_name,
                device.get("installationSite"),
                device.get("id"),
            )
            filename = "_".join(safe_filename_part(part) for part in parts)
            tasks.append((json_file, picture_url.strip(), output_dir / f"{filename}.jpg"))
    return tasks


def discover_json_files(input_path: Path) -> list[Path]:
    """根据输入参数返回要处理的 JSON 文件列表。"""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*.json") if not p.name.startswith("."))
    raise FileNotFoundError(f"找不到路径：{input_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量下载巡检设备图片（支持断点续传）。"
        "文件名为 deptName_name_deptName_installationSite_id.jpg。"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "数据集",
        help="JSON 文件或目录路径（默认扫描 数据集/ 目录下所有 JSON）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出目录（单个 JSON 时直接存入该目录；多个 JSON 时按 JSON 名各建子目录；"
        "默认与每个 JSON 同目录、以 JSON 同名子文件夹存放）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要下载的文件，不实际下载",
    )
    args = parser.parse_args()

    try:
        json_files = discover_json_files(args.input)
    except FileNotFoundError as error:
        print(f"{error}", file=sys.stderr)
        return 1

    if not json_files:
        print("没有找到任何 JSON 文件。", file=sys.stderr)
        return 1

    print(f"共发现 {len(json_files)} 个 JSON 文件。")
    tasks = collect_tasks(json_files, args.output)

    if not tasks:
        print("没有可下载的图片。")
        return 0

    if args.dry_run:
        for _, _, target in tasks:
            print(target)
        print(f"\n共找到 {len(tasks)} 张图片。")
        return 0

    # 按输出目录分组，提前创建
    dirs = {target.parent for _, _, target in tasks}
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    succeeded = skipped = failed = 0
    interrupted = False
    try:
        for index, (json_file, picture_url, target) in enumerate(tasks, start=1):
            try:
                # 已存在完整文件 -> 跳过
                if target.exists() and target.stat().st_size > 0:
                    skipped += 1
                    print(f"[{index}/{len(tasks)}] 已存在，跳过：{target.name}")
                    continue
                download_picture(picture_url, target)
                succeeded += 1
                print(f"[{index}/{len(tasks)}] 已下载：{target.name}")
            except KeyboardInterrupt:
                raise
            except (OSError, URLError, ValueError) as error:
                failed += 1
                print(
                    f"[{index}/{len(tasks)}] 最终下载失败：{target.name}（{error}）",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        interrupted = True

    print(
        f"\n{'已停止' if interrupted else '全部完成'}："
        f"成功 {succeeded}，跳过 {skipped}，失败 {failed} / 共 {len(tasks)} 张。"
    )
    if interrupted:
        print("提示：再次运行同一命令，未完成的 .part 文件将自动从断点继续。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
