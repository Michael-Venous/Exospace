import unittest
import cv2
import numpy as np
from brain.verification import visibly_changed
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

if __name__=='__main__':unittest.main()
