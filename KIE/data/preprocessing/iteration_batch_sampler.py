from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler


class IterationBatchSampler(Sampler):
    
    def __init__(self, batch_sampler: BatchSampler, start_iteration, max_iterations):
        """
        Wraps a BatchSampler, re-sampling from it until a specified number of iterations have been sampled.
        
        Args:
            batch_sampler: The batch sampler.
            start_iteration: The number of iterations to start with.
            max_iterations: The maximum number of iterations.
            
        """
        super(IterationBatchSampler, self).__init__(batch_sampler.sampler.data_source)
        self.batch_sampler = batch_sampler
        self.max_iterations = max_iterations
        self.start_iteration = start_iteration
    
    def __iter__(self):
        iteration = self.start_iteration
        while iteration <= self.max_iterations:
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.max_iterations:
                    break
                yield batch
    
    def __len__(self):
        return self.max_iterations
