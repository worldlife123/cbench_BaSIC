import os
import logging

from cbench.utils.logger import setup_logger
from cbench.utils.sync_utils import GeneralFileSyncUtils


class FileWriteLocker(object):
    def __init__(self, open_file, lock_file, *args, **kwargs) -> None:
        self.open_file = open_file
        self.lock_file = lock_file
        self.args = args
        self.kwargs = kwargs

        # self.lock_acquired = False
        self.file_obj = None

    def __enter__(self):
        if os.path.exists(self.lock_file):
            return None
        else:
            with open(self.lock_file, 'w'):
                # self.lock_acquired = True
                self.file_obj = open(self.open_file, *self.args, **self.kwargs)
                return self.file_obj

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file_obj is not None:
            self.file_obj.close()
            # NOTE: in multiprocessing, there may be multiple processes trying to delete the lock, 
            # so we need to catch exceptions here!
            try:
                if os.path.exists(self.lock_file):
                    os.remove(self.lock_file)
            except:
                pass


class BaseEngine(object):
    """ Engine objects are objects that interacts 
        with input (from stdin, filesystems, netio, etc)
        or output (to logger, filesystems, netio, etc).

        Currently only some variables are kept.

    Args:
        object (_type_): _description_
    """    
    def __init__(self, *args,
                 output_dir=None,
                 sync_url=None,
                 sync_dir=None,
                 sync_interval=0,
                 sync_utils_params=dict(),
                 sync_start_action="sync_directory",
                 sync_start_action_params=dict(),
                 sync_loop_action="upload_directory",
                 sync_loop_action_params=dict(),
                 sync_end_action="upload_directory",
                 sync_end_action_params=dict(),
                #  sync_upload_only=False,
                #  sync_upload_params=dict(),
                #  sync_download_params=dict(),
                 logger=None,
                 **kwargs):

        self.sync_utils = None
        
        self.setup_engine(*args,
            output_dir=output_dir,
            sync_url=sync_url,
            sync_dir=sync_dir,
            sync_interval=sync_interval,
            sync_utils_params=sync_utils_params,
            sync_start_action=sync_start_action,
            sync_start_action_params=sync_start_action_params,
            sync_loop_action=sync_loop_action,
            sync_loop_action_params=sync_loop_action_params,
            sync_end_action=sync_end_action,
            sync_end_action_params=sync_end_action_params,
            # sync_upload_only=sync_upload_only,
            # sync_upload_params=sync_upload_params,
            # sync_download_params=sync_download_params,
            logger=logger,
            **kwargs
        )

    def setup_engine(self, *args,
                 output_dir=None,
                 logger=None,
                 **kwargs):
        # output dir
        self.output_dir = output_dir
        if output_dir is not None:
            if not os.path.exists(output_dir):
                # NOTE: Sometimes FileExistsError is still thrown... dont know why...
                os.makedirs(output_dir, exist_ok=True)
            self.remove_lock_file()

        # global logger
        if logger is None:
            if output_dir:
                logger = setup_logger(self.__class__.__name__, outdir=self.output_dir, label="log")
            else:
                logger = setup_logger(self.__class__.__name__)
            logger.setLevel(logging.INFO)

        # if logger is not None:
        self.logger = logger
        
        self.engine_kwargs = kwargs


        # sync dir
        if self.output_dir is not None:
            
            # finish last sync task
            # self.stop_engine()
            if self.sync_utils is not None:
                # self.logger.info(f"File sync : upload start!")
                # self.sync_utils.upload_directory(self.remote_dir, self.output_dir, **self.engine_kwargs["sync_upload_params"])
                # self.logger.info(f"File sync : upload complete!")
                self._sync_action(self.engine_kwargs.get("sync_end_action"), **self.engine_kwargs.get("sync_end_action_params"))
                self.sync_utils.stop_all_timers()

            self.sync_utils = None
            # self.engine_kwargs["sync_url"] = sync_url
            # self.sync_dir = sync_dir
            # self.engine_kwargs["sync_interval"] = sync_interval
            # self.sync_start_action = sync_start_action
            # self.sync_upload_only = sync_upload_only
            # self.engine_kwargs["sync_upload_params"] = sync_upload_params
            # self.engine_kwargs["sync_download_params"] = sync_download_params
            self.sync_url = self.engine_kwargs.get("sync_url")
            self.sync_dir = self.engine_kwargs.get("sync_dir")
            if self.sync_url is not None:
                self.logger.info(f"File sync : Sync with url {self.sync_url} : {self.remote_dir} <--> {self.output_dir}")
                self.sync_utils = GeneralFileSyncUtils(self.sync_url, **self.engine_kwargs["sync_utils_params"])
                self._sync_action(self.engine_kwargs.get("sync_start_action"), self.engine_kwargs.get("sync_start_action_params"))
                # if not self.sync_upload_only:
                #     self.logger.info(f"File sync : download start!")
                #     self.sync_utils.download_directory(self.remote_dir, self.output_dir, **self.engine_kwargs["sync_download_params"])
                #     self.logger.info(f"File sync : download complete!")
                if self.engine_kwargs["sync_interval"] > 0:
                    self._sync_action(self.engine_kwargs.get("sync_loop_action"), loop=True, **self.engine_kwargs.get("sync_loop_action_params"))
                    # self.sync_utils.register_timer("upload_directory", self.remote_dir, self.output_dir, interval=self.engine_kwargs["sync_interval"], **self.engine_kwargs["sync_upload_params"])

    def _sync_action(self, action=None, loop=False, **kwargs):
        if action is not None:
            if loop:
                self.sync_utils.register_timer(action, self.remote_dir, self.output_dir, interval=self.engine_kwargs["sync_interval"], **kwargs)
            else:
                self.logger.info(f"File sync : {action} start!")
                getattr(self.sync_utils, action)(self.remote_dir, self.output_dir, **kwargs)
                self.logger.info(f"File sync : {action} complete!")

    def setup_engine_from_copy(self, other, **kwargs):
        if isinstance(other, BaseEngine):
            params = dict(
                output_dir=other.output_dir,
                # sync_url=other.sync_url,
                # sync_dir=other.sync_dir,
                # sync_interval=other.sync_interval,
                # sync_start_action=other.sync_start_action,
                # sync_upload_only=other.sync_upload_only,
                # sync_upload_params=other.sync_upload_params,
                # sync_download_params=other.sync_download_params,
                logger=other.logger,
                **other.engine_kwargs,
            )
            params.update(**kwargs)
            self.setup_engine(**params)

    def stop_engine(self):
        # sync dir
        # finish last sync task
        if self.sync_utils is not None:
            # self.logger.info(f"File sync : upload start!")
            # self.sync_utils.upload_directory(self.remote_dir, self.output_dir, **self.engine_kwargs["sync_upload_params"])
            # self.logger.info(f"File sync : upload complete!")
            self._sync_action(self.engine_kwargs.get("sync_end_action"), **self.engine_kwargs.get("sync_end_action_params"))
            self.sync_utils.stop_all_timers()

            self.sync_utils = None

        # self.engine_kwargs["sync_url"] = None
        # self.sync_dir = None

        # # output dir
        # self.output_dir = None

        # # logger
        # self.logger = None
        
    @property
    def remote_dir(self):
        return self.sync_dir if self.sync_dir is not None else self.output_dir

    def remove_lock_file(self, lock_file_name="lock.tmp"):
        lock_file_path = os.path.join(self.output_dir, lock_file_name)
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)

    def open_file_safe(self, *args, file_dir=None, sub_dir=None, lock_file_name="lock.tmp", **kwargs):
        if file_dir is None:
            assert sub_dir is not None
            file_dir = os.path.join(self.output_dir, sub_dir)
        # if self.acquire_lock_file():
        #     return open(os.path.join(self.output_dir, sub_dir))
        return FileWriteLocker(
            file_dir, 
            os.path.join(self.output_dir, lock_file_name), *args, **kwargs)
