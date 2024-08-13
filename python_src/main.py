# from loader.text_chunker import TextChunker
from storage.local_storage import LocalStorage
from eval.dataset import Dataset
from core.data_bundle import DataBundle

if __name__ == '__main__':
    """text_chunker test"""
    # document = 'Santa claus is coming to town. ' * 10
    
    # text = TextChunker().read_text_file('../examples/books/book1.txt')
    # chunks = TextChunker().get_text_chunks('book1.txt', text, 5)
    # for chunk in chunks:
    #     print('=' * 100)
    #     print(chunk)

    """eval(dataset) test"""
    storage = LocalStorage()

    def get_receipt_image(storage, task_id, receipt: dict) -> DataBundle:
        print(receipt['file'])
        image = storage.read_image(f'../examples/receipts/{receipt["file"]}')
        return DataBundle.from_image(image)

    def get_receipt_total(storage, task_id, receipt: dict) -> DataBundle:
        return DataBundle.from_float(receipt['total'])
    
    receipt_dataset = Dataset(
        storage.read_jsonl_file('../examples/eval_datasets/receipts.jsonl'),
        context_loader=get_receipt_image,
        ground_truth_loader=get_receipt_total,
        task_id='test_task_id'
    )
    
    for context, ground_truth, metadata, cached_result in receipt_dataset:
        print(context, ground_truth, metadata, cached_result)

    """eval(0)"""