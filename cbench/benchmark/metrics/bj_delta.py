import math
import numpy as np

from .base import BaseMetric

class BJDeltaMetric(BaseMetric):
    def __init__(self, reference_pts=None, collect_metric_names=("compressed_length", "psnr"), mode=0, **kwargs):
        """AI is creating summary for __init__

        Args:
            reference_pts ([type], optional): Base points for reference. Defaults to None.
            mode (int, optional): mode 0 : BD-PSNR, mode 1 : BD-Rate. Defaults to 0.
        """        
        super().__init__()
        self.reference_pts = reference_pts
        self.collect_metric_names = collect_metric_names
        self.mode = mode
        assert self.mode in [0,1]

    @property
    def name(self):
        name = self.metric_names[1] if self.mode==0 else "rate"
        return f"BD-{name}"

    @property
    def metric_names(self):
        return [self.name]

    def __call__(self, output, target=None):
        if target is None:
            target = self.reference_pts
        R1, PSNR1 = output
        R2, PSNR2 = target
        try:
            result = {
                self.name : bj_delta(R1, PSNR1, R2, PSNR2, mode=self.mode)
            }
        except:
            print("bj_delta calculation failed!")
            result = {
                self.name : -100,
            }
        self.metric_logger.update(**result)
        return result


# https://github.com/Anserw/Bjontegaard_metric
def bj_delta(R1, PSNR1, R2, PSNR2, mode=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # find integral
    if mode == 0:
        # least squares polynomial fit
        p1 = np.polyfit(lR1, PSNR1, 3)
        p2 = np.polyfit(lR2, PSNR2, 3)

        # integration interval
        min_int = max(min(lR1), min(lR2))
        max_int = min(max(lR1), max(lR2))

        # indefinite integral of both polynomial curves
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        # evaluates both poly curves at the limits of the integration interval
        # to find the area
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

        # find avg diff between the areas to obtain the final measure
        avg_diff = (int2-int1)/(max_int-min_int)
    else:
        # rate method: sames as previous one but with inverse order
        p1 = np.polyfit(PSNR1, lR1, 3)
        p2 = np.polyfit(PSNR2, lR2, 3)

        # integration interval
        min_int = max(min(PSNR1), min(PSNR2))
        max_int = min(max(PSNR1), max(PSNR2))

        # indefinite interval of both polynomial curves
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        # evaluates both poly curves at the limits of the integration interval
        # to find the area
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

        # find avg diff between the areas to obtain the final measure
        avg_exp_diff = (int2-int1)/(max_int-min_int)
        avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff