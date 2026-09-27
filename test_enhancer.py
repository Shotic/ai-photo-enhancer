import unittest
import numpy as np
from PIL import Image
from enhancer import ImageEnhancer

class TestImageEnhancer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enhancer = ImageEnhancer()

    def test_normal_case_4x(self):
        # 128x128 RGB image
        img = Image.new("RGB", (128, 128), color=(120, 180, 240))
        res = self.enhancer.enhance(img, target_scale="4x")
        self.assertEqual(res.size, (512, 512))
        self.assertEqual(res.mode, "RGB")

    def test_target_scale_2x_and_1x(self):
        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        res_2x = self.enhancer.enhance(img, target_scale="2x")
        self.assertEqual(res_2x.size, (200, 200))

        res_1x = self.enhancer.enhance(img, target_scale="1x (Enhanced Original Size)")
        self.assertEqual(res_1x.size, (100, 100))

    def test_rgba_transparency_preservation(self):
        # Create an RGBA image with a transparent quadrant
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        arr[:, :, :3] = 180
        arr[:50, :50, 3] = 0    # Transparent
        arr[50:, :, 3] = 255   # Opaque
        arr[:50, 50:, 3] = 128  # Semi-transparent
        img = Image.fromarray(arr)

        res = self.enhancer.enhance(img, target_scale="2x")
        self.assertEqual(res.size, (200, 200))
        self.assertEqual(res.mode, "RGBA")
        # Check alpha channel preserved at corner
        res_arr = np.array(res)
        self.assertEqual(res_arr[10, 10, 3], 0)
        self.assertEqual(res_arr[180, 180, 3], 255)

    def test_tiling_on_large_image(self):
        # 500x600 triggers tiling because tile_size is 384
        img = Image.new("RGB", (500, 600), color=(80, 140, 200))
        res = self.enhancer.enhance(img, target_scale="2x")
        self.assertEqual(res.size, (1000, 1200))

    def test_oversized_image_safety_clamp(self):
        # 2500x2000 image with max_input_dimension clamped to 1000
        img = Image.new("RGB", (2500, 2000), color=(100, 100, 100))
        res = self.enhancer.enhance(img, target_scale="1x (Enhanced Original Size)", max_input_dimension=1000)
        # 2500 scaled down to 1000, 2000 scaled down to 800
        self.assertEqual(res.size, (1000, 800))

    def test_none_input_error_handling(self):
        with self.assertRaises(ValueError):
            self.enhancer.enhance(None)

if __name__ == "__main__":
    unittest.main()
