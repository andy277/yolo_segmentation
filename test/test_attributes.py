import unittest
from core.detect import Segmentation
import logging

logging.basicConfig(filename="logs/log.txt",
                    level=logging.INFO, filemode="w",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',)
logger = logging.getLogger(__name__)

class MyTestCase(unittest.TestCase):
    def test_attributes(self):
        image_path = 'data/biker.jpg'

        segmentation = Segmentation()
        self.assertIsNotNone(segmentation)

        results = segmentation.segment_detect(image_path)[0]
        #Box attribute
        boxes = results.boxes
        self.assertIsNotNone(boxes)

        #original image
        orig_img = results.orig_img
        self.assertIsNotNone(orig_img)

        #original_shape
        orig_shape = results.orig_shape
        self.assertIsNotNone(orig_shape)

        #masks
        masks = results.masks
        self.assertIsNotNone(masks)


if __name__ == '__main__':
    unittest.main()