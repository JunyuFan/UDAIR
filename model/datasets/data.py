import torch
import math
from torch.utils.data import DataLoader
from model.datasets.dataset import get_train_set, get_eval_set, get_test_set
from torch.utils.data.sampler import RandomSampler

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, mini_batch_size):
        self.dataset = dataset
        self.batch_size = mini_batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # Resample smaller datasets so every task contributes evenly in each epoch.
        epoch_samples = self.largest_dataset_size * self.number_of_datasets
        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)


        return iter(final_samples_list)
    

def get_train_loader(dataset_path, tasks, patch_size, data_augmentation, batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True):
    
    train_dataset = get_train_set(dataset_path, tasks, patch_size, data_augmentation)

    return DataLoader(dataset=train_dataset,
                      sampler=BatchSchedulerSampler(train_dataset, batch_size//len(tasks)),
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      shuffle=False,
                      drop_last=drop_last)

def get_eval_loader(dataset_path, tasks, image_size, batch_size, num_workers=4, shuffle=False):
    val_dataset = get_eval_set(dataset_path, tasks, image_size=image_size)

    return DataLoader(dataset=val_dataset,
                    #   sampler=BatchSchedulerSampler(val_dataset, batch_size//len(tasks)),
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle)

def get_test_loader(dataset_path, test_dataset, image_size, batch_size, num_workers=4):
    test_dataset = get_test_set(dataset_path, test_dataset, image_size=image_size)

    return DataLoader(dataset=test_dataset, 
                      batch_size=batch_size, 
                      num_workers=num_workers, 
                      pin_memory=True, 
                    #   persistent_workers=True,
                      shuffle=False, 
                      drop_last=False)



