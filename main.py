import logging
import os
import time
from pathlib import Path

import yaml
from fsspec.implementations.sftp import SFTPFileSystem

from counter import Counter

if __name__ == '__main__':

    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    log = logging.getLogger(__name__)
    fs = SFTPFileSystem(cfg["sftp"]["host"], username=cfg["sftp"]["user"], key_filename=cfg["sftp"]["key_path"])
    directory = fs.ls("captures")
    for item in directory:
        if not Path(item).exists():
            logging.info("Downloading " + str(item))
            fs.get_file(item, item)
    for item in directory:
        start_time = time.time()
        counter = Counter(item, False)
        counter.calculate()
        logging.info("--- %s seconds ---" % (time.time() - start_time))
        logging.info("Deleting file" + str(item))
        fs.delete(item)
        os.remove(item)

