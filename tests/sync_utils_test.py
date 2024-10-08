import unittest

import os
import shutil
from configs.sync_utils import SyncUtils

SYNC_URL = "experiments"

class TestSyncUtils(unittest.TestCase):

    def test_file(self):
        sync_obj = SyncUtils()
        test_upload_file = "README.md"
        test_download_file = "test_file"
        sync_obj.upload(test_upload_file, test_upload_file)
        sync_obj.download(test_upload_file, test_download_file)

        with open(test_upload_file, 'r') as f:
            up_content = f.read()

        with open(test_download_file, 'r') as f:
            dn_content = f.read()
        
        os.remove(test_download_file)

        self.assertSequenceEqual(up_content, dn_content)

    def test_dir(self):
        sync_obj = SyncUtils()
        test_upload_dir = "tools"
        test_download_dir = "test_dir"
        sync_obj.upload_directory(test_upload_dir, test_upload_dir)
        sync_obj.download_directory(test_upload_dir, test_download_dir)

        # TODO: check content
        # self.assertSequenceEqual(up_content, dn_content)
        shutil.rmtree(test_download_dir)

        test_upload_dir = "tools"
        test_download_dir = "test_dir"
        sync_obj.sync_directory(test_upload_dir, test_upload_dir)
        sync_obj.sync_directory(test_upload_dir, test_download_dir)

        # TODO: check content
        # self.assertSequenceEqual(up_content, dn_content)
        shutil.rmtree(test_download_dir)

if __name__ == '__main__':
    unittest.main()