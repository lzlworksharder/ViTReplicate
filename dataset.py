from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import tensor,stack
def load_train_val_cifar100(data_dir,train_trans, test_trans,batch_size=64,num_workers=4):
    dataset = load_dataset(data_dir,split="train")
    dataset_dict = dataset.train_test_split(test_size=0.02)
    train_set = dataset_dict['train']
    val_set = dataset_dict['test']
    def train_trans_batch(examples):
        examples['pixel_values'] = [train_trans(x.convert('RGB')) for x in examples['img']]
        return examples
    def val_trans_batch(examples):
        examples['pixel_values'] = [test_trans(x.convert('RGB')) for x in examples['img']]
        return examples
    train_set.set_transform(train_trans_batch)
    val_set.set_transform(val_trans_batch)
    def collate_fn(batch):
        pixel_values = stack([x['pixel_values'] for x in batch])
        labels = tensor([x['fine_label'] for x in batch])
        return {'pixel_values': pixel_values, 'labels': labels}
    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn,
                              pin_memory=True,
                              num_workers=num_workers,
                              )
    val_loader = DataLoader(val_set, 
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            pin_memory=True,
                            num_workers=num_workers,
                            )
    return train_loader, val_loader

if __name__ == "__main__":
    from torchvision import transforms
    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    train_loader, val_loader = load_train_val_cifar100(
        "/root/python_projects/ViTReplay/data/cifar100", 
        train_trans, test_trans)

    print(next(iter(train_loader))['pixel_values'].shape)