import unittest
import random
import numpy as np
from cbench.tans import BufferedTansEncoder, TansEncoder, TansDecoder, create_ctable_using_cnt, create_dtable_using_cnt

class TestTansCoding(unittest.TestCase):

    def test_tans_coding(self):
        rnd_cnt = np.random.randint(1, 100, (8, 256))
        data = np.random.randint(0, 256, (2, 3, 32, 32))
        indexes = np.random.randint(0, 8, (2, 3, 32, 32))

        ctables = create_ctable_using_cnt(rnd_cnt)
        dtables = create_dtable_using_cnt(rnd_cnt)
        
        encoder = TansEncoder()
        decoder = TansDecoder()
        byte_string = encoder.encode_with_indexes(
            data.astype(np.int32).reshape(-1).tolist(), indexes.astype(np.int32).reshape(-1).tolist(),
            ctables,
            [0]*len(ctables),
        )
        decoded_data = decoder.decode_with_indexes(
            byte_string, indexes.astype(np.int32).reshape(-1).tolist(),
            dtables,
            [0]*len(dtables),
        )
        self.assertSequenceEqual(data.reshape(-1).tolist(), decoded_data)

        # one time
        # encoder = TansEncoder()
        # decoder = TansDecoder()
        # byte_string = encoder.encode_with_indexes_np(
        #     data.astype(np.int32), indexes.astype(np.int32),
        #     ctables,
        #     np.zeros(len(ctables)),
        # )
        # decoded_data = decoder.decode_with_indexes_np(
        #     byte_string, indexes.astype(np.int32),
        #     dtables,
        #     np.zeros(len(dtables)),
        # )
        # self.assertSequenceEqual(data.tolist(), decoded_data.tolist())

        # streaming
        encoder = TansEncoder()
        encoder.create_ctable_using_cnt(rnd_cnt)
        decoder = TansDecoder()
        decoder.create_dtable_using_cnt(rnd_cnt)
        byte_string = encoder.encode_with_indexes_np(
            data.astype(np.int32), indexes.astype(np.int32),
        )
        decoded_data = decoder.decode_with_indexes_np(
            byte_string, indexes.astype(np.int32),
        )
        self.assertSequenceEqual(data.tolist(), decoded_data.tolist())

    def test_tans_autoregressive_coding(self):
        rnd_cnt = np.random.randint(1, 100, (8, 256, 256, 256))
        # data = np.random.randint(0, 256, (2, 3, 32, 32))
        data = np.arange(2*3*4*4).reshape(2, 3, 4, 4)
        indexes = np.random.randint(0, 8, (2, 3, 4, 4))
        ar_offset = np.array([[0, -1, 0], [0, 0, -1]])

        ctables = create_ctable_using_cnt(rnd_cnt)
        dtables = create_dtable_using_cnt(rnd_cnt)
        
        # one time
        # encoder = TansEncoder()
        # decoder = TansDecoder()
        # byte_string = encoder.encode_autoregressive_np(
        #     data.astype(np.int32), indexes.astype(np.int32),
        #     ar_offset,
        #     ctables,
        #     [0]*len(ctables), 255
        # )
        # decoded_data = decoder.decode_autoregressive_np(
        #     byte_string, indexes.astype(np.int32),
        #     ar_offset,
        #     dtables,
        #     [0]*len(dtables), 255
        # )
        # self.assertSequenceEqual(data.reshape(-1).tolist(), decoded_data.reshape(-1).tolist())

        # streaming
        encoder = TansEncoder()
        encoder.create_ctable_using_cnt(rnd_cnt)
        print(rnd_cnt.shape)
        decoder = TansDecoder()
        decoder.create_dtable_using_cnt(rnd_cnt)
        byte_string = encoder.encode_autoregressive_np(
            data.astype(np.int32), indexes.astype(np.int32),
            ar_offset,
        )
        decoded_data = decoder.decode_autoregressive_np(
            byte_string, indexes.astype(np.int32),
            ar_offset,
        )
        self.assertSequenceEqual(data.tolist(), decoded_data.tolist())

        
if __name__ == '__main__':
    unittest.main()