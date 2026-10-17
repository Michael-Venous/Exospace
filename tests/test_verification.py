import unittest
import cv2
import numpy as np
from brain.verification import highlight_rows, visibly_changed
from brain.capture import thumbnail, frame_change

class VerificationTests(unittest.TestCase):
    def test_small_form_edit_counts_as_progress(self):
        a=np.full((800,1280,3),245,np.uint8)
        b=a.copy()
        cv2.putText(b,'19:30',(700,400),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
        self.assertLess(frame_change(thumbnail(a),thumbnail(b)),1.5)
        self.assertTrue(visibly_changed(a,b))

    def test_identical_frame_and_isolated_noise_are_not_progress(self):
        a=np.full((800,1280,3),245,np.uint8)
        self.assertFalse(visibly_changed(a,a.copy()))
        b=a.copy();b[200,300]=0
        self.assertFalse(visibly_changed(a,b))

    def test_resolution_change_is_visible(self):
        self.assertTrue(visibly_changed(np.zeros((800,1280,3),np.uint8),np.zeros((600,900,3),np.uint8)))

    def test_highlight_rows_finds_the_bar_a_key_moved_onto(self):
        a=np.full((240,320,3),152,np.uint8);a[40:60,20:200]=0
        b=np.full((240,320,3),152,np.uint8);b[64:84,20:200]=0
        self.assertEqual(highlight_rows(a,b),(64,83))
        c=a.copy();cv2.putText(c,'19:30',(100,150),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
        self.assertIsNone(highlight_rows(a,c))

if __name__=='__main__':unittest.main()
