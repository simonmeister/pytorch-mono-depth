import os
import zipfile
import sys
from urllib.request import FancyURLopener


def download(url, destination, tmp_dir='/tmp'):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\rDownloading %s %.1f%%' % (url,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    urlretrieve = FancyURLopener().retrieve
    if url.endswith('.zip'):
        local_zip_path = os.path.join(tmp_dir, 'datasets_download.zip')
        urlretrieve(url, local_zip_path, _progress)
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(local_zip_path)
    else:
        urlretrieve(url, destination, _progress)


def maybe_download(url, destination):
    if not os.path.isfile(destination):
        download(url, destination)


def transform_chw(transform, lst):
    """Convert each array in lst from CHW to HWC"""
    return transform([x.transpose((1, 2, 0)) for x in lst])
