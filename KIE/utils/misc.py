import time
import torch
import datetime as dt


def get_process_time(start_time, current_iteration, max_iterations):
    """
    Calculate the elapsed, remaining and ETA times.
    
    Args:
        start_time: The starting time.
        current_iteration: The current iteration.
        max_iterations: The maximum number of iterations.

    Returns:
        A tuple containing the elapsed, remaining and the ETA times.
        
    """
    elapsed_time = time.time() - start_time
    
    estimated_time = (elapsed_time / current_iteration) * max_iterations
    remaining_time = estimated_time - elapsed_time  # in seconds
    
    finishtime = str(dt.datetime.fromtimestamp(start_time + estimated_time).strftime("%Y/%m/%d at %H:%M:%S"))
    
    times = (int(elapsed_time), int(remaining_time), finishtime)
    
    return times


# Adapted from https://github.com/pytorch/examples/blob/adc5bb40f1fa5ebae690787b474af4619df170b8/imagenet/main.py#L363
class AverageMeter(object):
    
    def __init__(self, fmt=":f"):
        """
        Computes and stores the average and current value.
        
        Args:
            fmt: The string format.
            
        """
        self.fmt = fmt
        self.count = 0
        self.value = 0.0
        self.total = 0.0
        self.global_avg = 0.0
    
    def update(self, value, n=1):
        """
        Update the current values.
        
        Args:
            value: The value to update with.
            n: A multiplier.
            
        """
        self.value = value
        self.total += (value * n)
        self.count += n
        self.global_avg = self.total / self.count
    
    def __str__(self):
        fmtstr = '{value' + self.fmt + '} ({global_avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
