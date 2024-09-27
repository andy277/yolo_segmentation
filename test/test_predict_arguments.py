import unittest
from core.detect import Segmentation
import logging

logging.basicConfig(filename="logs/log.txt",
                    level=logging.INFO, filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',)
logger = logging.getLogger(__name__)


class MyTestCase(unittest.TestCase):
    def test_predict(self):
        image_path = 'data/biker.jpg'

        segmentation = Segmentation()
        self.assertIsNotNone(segmentation)

        results = segmentation.segment_detect_call_predict(image_path, show=True, show_conf=True)
        self.assertIsNotNone(results)

        logger.info('test_predict_argument run')


if __name__ == '__main__':
    unittest.main()
