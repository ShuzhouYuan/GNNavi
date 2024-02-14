import os.path

from datasets import load_dataset, load_from_disk

ROOT_FOLEDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_from_local(task_name, splits):
    dataset_path = os.path.join(ROOT_FOLEDER, 'datasets', task_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset_path: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    dataset = [dataset[split] for split in splits]
    return dataset


def load_huggingface_dataset_train_and_test(task_name):
    dataset = None
    if task_name == 'sst2':
        try:
            dataset = load_from_local(task_name, ['train', 'validation'])
        except FileNotFoundError:
            dataset = load_dataset('sst2', split=['train', 'validation'])
        for i, _ in enumerate(dataset):
            dataset[i] = dataset[i].rename_column('sentence', 'text')
        # rename validation to test
    elif task_name == 'agnews':
        try:
            dataset = load_from_local(task_name, ['train', 'test'])
        except FileNotFoundError:
            dataset = load_dataset('ag_news', split=['train', 'test'])
    elif task_name == 'trec':
        try:
            dataset = load_from_local(task_name, ['train', 'test'])
        except FileNotFoundError:
            dataset = load_dataset('trec', split=['train', 'test'])
        coarse_label_name = 'coarse_label' if 'coarse_label' in dataset[
            0].column_names else 'label-coarse'
        for i, _ in enumerate(dataset):
            dataset[i] = dataset[i].rename_column(coarse_label_name, 'label')
    elif task_name == 'emo':
        try:
            dataset = load_from_local(task_name, ['train', 'test'])
        except FileNotFoundError:
            dataset = load_dataset('emo', split=['train', 'test'])
    elif task_name == 'amazon':
        try:
            dataset = load_from_local(task_name, ['train', 'test'])
        except FileNotFoundError:
            dataset = load_dataset('amazon_polarity', split=['train', 'test'])
        for i, _ in enumerate(dataset):
            dataset[i] = dataset[i].rename_column('content', 'text')
    elif task_name == 'yahoo':
        try:
            dataset = load_from_local(task_name, ['train', 'test'])
        except FileNotFoundError:
            dataset = load_dataset('yahoo_answers_topics', split=['train', 'test'])
        for i, _ in enumerate(dataset):
            dataset[i] = dataset[i].rename_column('question_title', 'text')
            dataset[i] = dataset[i].rename_column('topic', 'label')
    elif task_name == 'hate':
        try:
            dataset = load_from_local(task_name, ['train'])
        except FileNotFoundError:
            dataset = load_dataset('hate_speech_offensive', split=['train'])
        for i, _ in enumerate(dataset):
            dataset[i] = dataset[i].rename_column('tweet', 'text')
            dataset[i] = dataset[i].rename_column('class', 'label')
        dataset = dataset[0].train_test_split(test_size=0.3)

    if dataset is None:
        raise NotImplementedError(f"task_name: {task_name}")
    dataset = {'train': dataset[0], 'test': dataset[1]} if task_name != 'hate' else {'train': dataset['train'], 'test': dataset['test']}
    return dataset

if __name__ == '__main__':
    dataset = load_huggingface_dataset_train_and_test('sst2')
    #dataset = load_dataset('amazon_reviews_multi', 'en', split=['train', 'test'],cache_dir='/home/hk-project-gnn4nlp/nu4126/.cache/huggingface/datasets')
   
    print(dataset)
