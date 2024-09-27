import base64
import logging
import time
import os, shutil
from pathlib import Path
import tempfile
import zlib
from typing import Dict, Tuple

import multiprocessing # implement multiprocess download/upload
import functools
import threading # implement async timer

OSS2_ENABLED = True
try:
    import oss2
    from oss2.utils import Crc64, make_crc_adapter
except:
    print("oss2 not installed! Disabling OSSUtils!")
    OSS2_ENABLED = False

from cbench.utils.logger import setup_logger


class FileSyncObject(object):
    def __init__(self, file_sync_utils, func_name, *args, interval=600, loop=True, logger=None, **kwargs) -> None:
        self.file_sync_utils = file_sync_utils
        self.func_name = func_name
        self.interval = interval
        self.loop = loop
        self.logger = logger
        
        self.args = args
        self.kwargs = kwargs

        self.exit_event = threading.Event()

    def request_exit(self):
        self.exit_event.set()

    # multiprocessing enter point
    def __call__(self, *args, **kwargs) -> None:
        while True:
            if self.exit_event.is_set(): break
            kwargs.update(**self.kwargs)
            getattr(self.file_sync_utils, self.func_name)(*args, *self.args, **kwargs)
            if not self.loop:
                break
            # allow exit event when sleeping
            for i in range(int(self.interval)):
                if self.exit_event.is_set(): 
                    return
                time.sleep(1)
            # await asyncio.sleep(self.interval)


class FileSyncUtilsInterface(object):
    def __init__(self, 
        remote_root="", # DEFAULT_SYNC_DISK_ROOT_PATH,
        max_retry=-1,
        num_process=0,
        logger=None,
        **kwargs
    ) -> None:
        self.remote_root = remote_root
        self.max_retry = max_retry
        # NOTE: pool cannot be pickled
        self.num_process = num_process
        if num_process > 0:
            self.pool = multiprocessing.Pool(num_process)

        self.logger = setup_logger("FileSyncUtilsInterface") if logger is None else logger
        self.kwargs = kwargs
        
        self.timers : Dict[str, Tuple[FileSyncObject, threading.Thread]] = dict()

        self._connect_to_remote()

    def register_timer(self, func_name, *args, interval=600, loop=True, **kwargs):
        func = getattr(self, func_name)
        if func is not None:
            sync_obj = FileSyncObject(self, func_name, *args, interval=interval, loop=loop, **kwargs)
            self.timers[func_name] = (sync_obj, threading.Thread(target=sync_obj))
            self.timers[func_name][1].start()

    def stop_all_timers(self):
        for name, timer in self.timers.items():
            timer[0].request_exit()
            timer[1].join(timeout=timer[0].interval)
        self.timers = dict()

    def _clone_wo_pool(self):
        return self.__class__(
            remote_root=self.remote_root,
            max_retry=self.max_retry,
            num_process=0, # no pool
            **self.kwargs
        )

    def _connect_to_remote(self):
        raise NotImplementedError()

    def _checksum_local_file(self, local_path):
        # TODO:
        raise NotImplementedError()
        # adapter = make_crc_adapter(file_obj)
        # adapter.read()
        # return adapter.crc

    def _checksum_remote_file(self, remote_path):
        raise NotImplementedError()
        # adapter = make_crc_adapter(file_obj)
        # adapter.read()
        # return adapter.crc

    def _compare_checksum(self, remote_path, local_path):
        return self._checksum_local_file(local_path) == self._checksum_remote_file(remote_path)

    def _iter_local_dir(self, local_dir : str):
        """ Iterate relative paths of local files

        Args:
            local_dir (str)

        Yields:
            str: relative file paths to local_dir
        """        
        for dirpath, dirnames, filenames in os.walk(local_dir):
            for fname in filenames:
                local_path = os.path.join(dirpath, fname)
                filename = os.path.relpath(local_path, local_dir)
                yield filename

    def _iter_remote_dir(self, remote_dir : str):
        """ Iterate relative paths of remote files.

        Args:
            remote_dir (str): _description_

        Yields:
            str: relative file paths to remote_dir
        """
        raise NotImplementedError()
        # yield from self._iter_local_dir(remote_dir)

    def _upload_to_dir_by_relpath(self, remote_dir, local_dir, fpath):
        local_path = os.path.join(local_dir, fpath)
        remote_path = os.path.join(remote_dir, fpath)
        self.upload(remote_path, local_path)

    def _download_to_dir_by_relpath(self, remote_dir, local_dir, fpath):
        local_path = os.path.join(local_dir, fpath)
        remote_path = os.path.join(remote_dir, fpath)
        self.download(remote_path, local_path)

    def _delete_local_by_relpath(self, local_dir, fpath):
        local_path = os.path.join(local_dir, fpath)
        self.local_delete(local_path)

    def _delete_remote_by_relpath(self, remote_dir, fpath):
        remote_path = os.path.join(remote_dir, fpath)
        self.remote_delete(remote_path)

    def local_exists(self, local_path) -> bool:
        return os.path.exists(local_path)

    def remote_exists(self, remote_path) -> bool:
        raise NotImplementedError()

    def local_delete(self, local_path):
        os.remove(local_path)

    def remote_delete(self, remote_path):
        raise NotImplementedError()
    
    def _download_from_remote(self, remote_path, local_path, *args, **kwargs):
        raise NotImplementedError()

    def download(self, remote_path, local_path, *args, checksum=True, allow_overwrite=True, **kwargs):
        retry_cnt = 0
        while retry_cnt < self.max_retry or self.max_retry < 0:
            try:
                if not self.remote_exists(remote_path):
                    self.logger.warning("Remote file {} not exist!".format(remote_path))
                    return
                if os.path.exists(local_path):
                    if not allow_overwrite:
                        self.logger.info("Local file exists! Skip downloading remote file {} to {}".format(remote_path, local_path))
                        return
                    elif checksum and self._compare_checksum(remote_path, local_path):
                        self.logger.info("Same checksum! Skip downloading remote file {} to {}".format(remote_path, local_path))
                        return
                # download
                self.logger.info("Download remote file {} to {}".format(remote_path, local_path))
                local_dir = os.path.dirname(os.path.abspath(local_path))
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir, exist_ok=True)
                self._download_from_remote(remote_path, local_path, *args, **kwargs)
                return
            except:
                # retry uploading if failed
                retry_cnt += 1
                self.logger.info("Retry Connection [{}/{}]...".format(retry_cnt, self.max_retry))
                self._connect_to_remote()
        self.logger.info("Download failed!")

    def download_directory(self, remote_dir, local_dir, *args,
        force_overwrite_dir=False,
        **kwargs):
        download_files = list(self._iter_remote_dir(remote_dir))
        if force_overwrite_dir:
            # TODO: this may be dangerous if downloading fails!
            # find local files that does not exist on oss and delete
            local_files = list(self._iter_local_dir(remote_dir))
            delete_files = set(local_files) - set(download_files)
            for fpath in delete_files:
                self._delete_local_by_relpath(remote_dir, fpath)

        # download
        if self.num_process > 0:
            new_obj = self._clone_wo_pool()
            for idx, _ in enumerate(
                self.pool.imap_unordered(functools.partial(new_obj._download_to_dir_by_relpath, remote_dir, local_dir), download_files)
            ):
                self.logger.info(f"Download process: {idx}/{len(download_files)}")
        else:
            for idx, filename in enumerate(download_files):
                self.logger.info(f"Download process: {idx}/{len(download_files)}")
                self._download_to_dir_by_relpath(remote_dir, local_dir, filename)

    def _upload_to_remote(self, remote_path, local_path, *args, **kwargs):
        raise NotImplementedError()

    def upload(self, remote_path, local_path, *args, checksum=True, allow_overwrite=True, **kwargs):
        if not os.path.isfile(local_path):
            self.logger.warning("local file {} not exist!".format(local_path))
            return
        retry_cnt = 0
        while retry_cnt < self.max_retry or self.max_retry < 0:
            try:
                # check if object exists
                if self.remote_exists(remote_path):
                    if not allow_overwrite:
                        self.logger.info("Remote File exists! Skip uploading remote file {} from {}".format(remote_path, local_path))
                        return
                    elif checksum and self._compare_checksum(remote_path, local_path):
                        self.logger.info("Same checksum! Skip uploading remote file {} from {}".format(remote_path, local_path))
                        return
                # uploading
                self.logger.info("Upload remote file {} from {}".format(remote_path, local_path))
                self._upload_to_remote(remote_path, local_path, *args, **kwargs)
                return
            except oss2.exceptions.RequestError:
                # retry uploading if failed
                retry_cnt += 1
                self.logger.info("Retry Connection [{}/{}]...".format(retry_cnt, self.max_retry))
                self._connect_to_remote()
        self.logger.info("Upload failed!")

    # TODO: handle multiple client upload to the same directory case
    def upload_directory(self, remote_dir, local_dir, *args, 
        force_overwrite_dir=False,
        snapshot_local_dir=True, # snapshot dir to avoid local file change during upload
        **kwargs):
        if snapshot_local_dir:
            tmpdir = tempfile.TemporaryDirectory()
            # copy everything in local dir to tmp as snapshot
            snapshot_dir = os.path.join(tmpdir.name, "snapshot")
            shutil.copytree(local_dir, snapshot_dir)
            local_dir = snapshot_dir
        upload_files = list(self._iter_local_dir(local_dir))
        if force_overwrite_dir:
            # find oss files that does not exist on local and delete
            remote_files = list(self._iter_remote_dir(remote_dir))
            delete_files = set(remote_files) - set(upload_files)
            for fpath in delete_files:
                self._delete_remote_by_relpath(remote_dir, fpath)
        
        # upload
        if self.num_process > 0:
            new_obj = self._clone_wo_pool()
            for idx, _ in enumerate(
                self.pool.imap_unordered(functools.partial(new_obj._upload_to_dir_by_relpath, remote_dir, local_dir), upload_files)
            ):
                self.logger.info(f"Upload process: {idx}/{len(upload_files)}")
        else:
            for idx, filename in enumerate(upload_files):
                self.logger.info(f"Upload process: {idx}/{len(upload_files)}")
                self._upload_to_dir_by_relpath(remote_dir, local_dir, filename)

        # for dirpath, dirnames, filenames in os.walk(local_dir):
        #     for fname in filenames:
        #         local_path = os.path.join(dirpath, fname)
        #         remote_path = os.path.join(remote_dir, os.path.relpath(local_path, local_dir))
        #         self.upload(remote_path, local_path)

        if snapshot_local_dir:
            tmpdir.cleanup()

    # def download_archive_and_extract(self, remote_path, local_path, *args, **kwargs):
    #     self.download(remote_path, local_path, *args, **kwargs)
    #     extract_archive(local_path)

    def _diff_directory(self, remote_dir, local_dir, *args, **kwargs):
        upload_files, download_files = [], []

        # check files on remote and local
        local_files = list(self._iter_local_dir(local_dir))
        remote_files = list(self._iter_remote_dir(remote_dir))
        # print(remote_dir, remote_files)

        # diff 2 lists
        upload_files = set(local_files) - set(remote_files)
        download_files = set(remote_files) - set(local_files)

        return upload_files, download_files

    def sync_file(self, remote_path, local_path, *args, **kwargs):
        if os.path.exists(local_path):
            if not self.remote_exists(remote_path):
                self.upload(remote_path, local_path, *args, **kwargs)
        else:
            self.download(remote_path, local_path, *args, **kwargs)

    def sync_directory(self, remote_dir, local_dir, *args,
        # check_hash=False,
        **kwargs):
        upload_files, download_files = self._diff_directory(
            remote_dir, local_dir, *args, **kwargs
        )

        # perform sync files
        if self.num_process > 0:
            new_obj = self._clone_wo_pool()
            for idx, _ in enumerate(
                self.pool.imap_unordered(functools.partial(new_obj._upload_to_dir_by_relpath, remote_dir, local_dir), upload_files)
            ):
                self.logger.info(f"Upload process: {idx}/{len(upload_files)}")
            for idx, _ in enumerate(
                self.pool.imap_unordered(functools.partial(new_obj._download_to_dir_by_relpath, remote_dir, local_dir), download_files)
            ):
                self.logger.info(f"Download process: {idx}/{len(download_files)}")
        else:
            for idx, filename in enumerate(upload_files):
                self.logger.info(f"Upload process: {idx}/{len(upload_files)}")
                self._upload_to_dir_by_relpath(remote_dir, local_dir, filename)
            for idx, filename in enumerate(download_files):
                self.logger.info(f"Download process: {idx}/{len(download_files)}")
                self._download_to_dir_by_relpath(remote_dir, local_dir, filename)

        

class DiskSyncUtils(FileSyncUtilsInterface):
    def _connect_to_remote(self):
        # We assume that the disk is already mounted!
        assert os.path.exists(self.remote_root), f"Directory {self.remote_root} not found!"

    def _checksum_local_file(self, local_path):
        # TODO: more efficient checksum?
        with open(local_path, 'rb') as f:
            return zlib.crc32(f.read())

    def _checksum_remote_file(self, remote_path):
        remote_file = os.path.join(self.remote_root, remote_path)
        return self._checksum_local_file(remote_file)

    def _iter_remote_dir(self, remote_dir : str):
        remote_dir_path = os.path.join(self.remote_root, remote_dir)
        yield from self._iter_local_dir(remote_dir_path)

    def remote_exists(self, remote_path) -> bool:
        remote_file = os.path.join(self.remote_root, remote_path)
        return self.local_exists(remote_file)

    def remote_delete(self, remote_path):
        remote_file = os.path.join(self.remote_root, remote_path)
        return self.local_delete(remote_file)

    def _download_from_remote(self, remote_path, local_path, *args, **kwargs):
        remote_file = os.path.join(self.remote_root, remote_path)
        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir): os.makedirs(local_dir)
        shutil.copyfile(remote_file, local_path, *args, **kwargs)

    def _upload_to_remote(self, remote_path, local_path, *args, **kwargs):
        remote_file = os.path.join(self.remote_root, remote_path)
        remote_dir = os.path.dirname(remote_file)
        if not os.path.exists(remote_dir): os.makedirs(remote_dir)
        shutil.copyfile(local_path, remote_file, *args, **kwargs)


class SSHSyncUtils(FileSyncUtilsInterface):
    # TODO:
    pass


class OSSUtils(FileSyncUtilsInterface):
    def __init__(self, 
        keyId_b64="", # DEFAULT_OSS_KEYID_BASE64, 
        keySec_b64="", # DEFAULT_OSS_KEYSEC_BASE64, 
        endpoint="", # DEFAULT_OSS_ENDPOINT, 
        bucket_name="", # DEFAULT_OSS_BUCKET_NAME,
        remote_root="", # DEFAULT_OSS_PERSONAL_ROOT,
        max_retry=-1,
        num_process=0,
        logger=None,
        **kwargs
    ) -> None:
        self.keyId_b64 = keyId_b64
        self.keyId = base64.b64decode(keyId_b64).decode('utf-8')
        self.keySec_b64 = keySec_b64
        self.keySec = base64.b64decode(keySec_b64).decode('utf-8')
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        super().__init__(remote_root=remote_root, max_retry=max_retry, num_process=num_process, logger=logger, **kwargs)

    def _clone_wo_pool(self):
        return self.__class__(
            keyId_b64=self.keyId_b64, 
            keySec_b64=self.keySec_b64, 
            endpoint=self.endpoint, 
            bucket_name=self.bucket_name,
            remote_root=self.remote_root,
            max_retry=self.max_retry,
            num_process=0, # no pool
        )

    def _connect_to_remote(self):
        if OSS2_ENABLED:
            auth = oss2.Auth(self.keyId, self.keySec)
            self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

    def _checksum_local_file(self, local_path):
        with open(local_path, 'rb') as f:
            adapter = make_crc_adapter(f.read())
            adapter.read()
            return adapter.crc

    def _checksum_remote_file(self, remote_path):
        remote_file = os.path.join(self.remote_root, remote_path)
        return self.bucket.get_object(remote_file).server_crc

    def _iter_remote_dir(self, remote_dir : str):
        """ Iterate relative paths of oss files

        Args:
            remote_dir (str): _description_

        Yields:
            str: relative file paths to remote_dir
        """        
        oss_prefix = os.path.join(self.remote_root, remote_dir) 
        if not oss_prefix.endswith("/"): oss_prefix += "/"
        for obj in oss2.ObjectIterator(self.bucket, prefix=oss_prefix):
            remote_path = os.path.relpath(obj.key, self.remote_root)
            filename = os.path.relpath(remote_path, remote_dir)
            yield filename

    def remote_exists(self, remote_path) -> bool:
        remote_file = os.path.join(self.remote_root, remote_path)
        return self.bucket.object_exists(remote_file)

    def remote_delete(self, remote_path):
        remote_file = os.path.join(self.remote_root, remote_path)
        self.bucket.delete_object(remote_file)

    def _download_from_remote(self, remote_path, local_path, *args, **kwargs):
        remote_file = os.path.join(self.remote_root, remote_path)
        self.bucket.get_object_to_file(remote_file, local_path, *args, **kwargs)

    def _upload_to_remote(self, remote_path, local_path, *args, **kwargs):
        remote_file = os.path.join(self.remote_root, remote_path)
        self.bucket.put_object_from_file(remote_file, local_path, *args, **kwargs)

    # TODO: add specific exception catching instead of overriding main func!
    def download(self, remote_path, local_path, *args, checksum=True, allow_overwrite=True, **kwargs):
        remote_file = os.path.join(self.remote_root, remote_path)
        retry_cnt = 0
        while retry_cnt < self.max_retry or self.max_retry < 0:
            try:
                if not self.bucket.object_exists(remote_file):
                    self.logger.warning("oss file {} not exist!".format(remote_path))
                    return
                if os.path.exists(local_path):
                    if not allow_overwrite:
                        self.logger.info("Local file exists! Skip downloading oss file {} to {}".format(remote_path, local_path))
                        return
                    elif checksum and self._compare_checksum(remote_path, local_path):
                        self.logger.info("Same checksum! Skip downloading oss file {} to {}".format(remote_path, local_path))
                        return
                # download
                self.logger.info("Download oss file {} to {}".format(remote_path, local_path))
                local_dir = os.path.dirname(os.path.abspath(local_path))
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir, exist_ok=True)
                self.bucket.get_object_to_file(remote_file, local_path, *args, **kwargs)
                return
            except oss2.exceptions.RequestError:
                # retry uploading if failed
                retry_cnt += 1
                self.logger.info("Retry Connection [{}/{}]...".format(retry_cnt, self.max_retry))
                self._connect_to_remote()
        self.logger.info("Download failed!")

    # TODO: add specific exception catching instead of overriding main func!
    def upload(self, remote_path, local_path, *args, checksum=True, allow_overwrite=True, **kwargs):
        if not os.path.isfile(local_path):
            self.logger.warning("local file {} not exist!".format(local_path))
            return
        remote_file = os.path.join(self.remote_root, remote_path)
        retry_cnt = 0
        while retry_cnt < self.max_retry or self.max_retry < 0:
            try:
                # check if object exists
                if self.bucket.object_exists(remote_file):
                    if not allow_overwrite:
                        self.logger.info("OSS File exists! Skip uploading oss file {} from {}".format(remote_path, local_path))
                        return
                    elif checksum and self._compare_checksum(remote_path, local_path):
                        self.logger.info("Same checksum! Skip uploading oss file {} from {}".format(remote_path, local_path))
                        return
                # uploading
                self.logger.info("Upload oss file {} from {}".format(remote_path, local_path))
                self.bucket.put_object_from_file(remote_file, local_path, *args, **kwargs)
                return
            except oss2.exceptions.RequestError:
                # retry uploading if failed
                retry_cnt += 1
                self.logger.info("Retry Connection [{}/{}]...".format(retry_cnt, self.max_retry))
                self._connect_to_remote()
        self.logger.info("Upload failed!")


def parse_url(url, *args, **kwargs):
    if url.startswith("ssh:"):
        # TODO: extract info from url!
        return SSHSyncUtils(*args, remote_root=url[4:], **kwargs)
    elif url.startswith("oss:"):
        # TODO: extract info from url!
        return OSSUtils(*args, remote_root=url[4:], **kwargs)
    else:
        return DiskSyncUtils(*args, remote_root=url, **kwargs)


class GeneralFileSyncUtils(object):
    def __init__(self, url : str, *args, **kwargs) -> None:
        self.impl : FileSyncUtilsInterface = parse_url(url, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.impl, name)


# import torch.utils.data
# class OSSDatasetWrapper(torch.utils.data._DatasetKind):
#     def __init__(self, dataset) -> None:
#         pass


# run this file to sync files
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", '-o', type=str, default="download")
    parser.add_argument("--url", '-u', type=str, default=DEFAULT_SYNC_URL)
    parser.add_argument("--remote_dir", '-r', type=str, default="experiments")
    parser.add_argument("--local_dir", '-l', type=str, default="experiments")
    parser.add_argument("--sync_dir", '-s', type=str)
    parser.add_argument("--allow_delete", '-d', action='store_true')
    parser.add_argument("--sync_interval", '-si', type=int, default=0)
    parser.add_argument("--num-process", '-p', type=int, default=0)

    args = parser.parse_args()

    # logger = setup_logger("OSSUtils", log_level='WARNING', log_level_file='INFO')
    sync_obj = GeneralFileSyncUtils(args.url, num_process=args.num_process)
    sync_local_dir = args.sync_dir if args.sync_dir else args.local_dir # "experiments"
    sync_remote_dir = args.sync_dir if args.sync_dir else args.remote_dir # "experiments"
    # oss.sync_directory(sync_remote_dir, sync_local_dir)
    while True:
        if args.op == "download":
            sync_obj.download_directory(sync_remote_dir, sync_local_dir, force_overwrite_dir=args.allow_delete)
        elif args.op == "upload":
            sync_obj.upload_directory(sync_remote_dir, sync_local_dir, force_overwrite_dir=args.allow_delete)
        elif args.op == "sync":
            sync_obj.sync_directory(sync_remote_dir, sync_local_dir)
        else:
            raise KeyError(f"Unknown op {args.op}")

        if args.sync_interval > 0:
            print(f"Sleeping {args.sync_interval} seconds...")
            time.sleep(args.sync_interval)
        else:
            break