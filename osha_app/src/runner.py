from model import ExplainNLP, Predictor, ManualChunker
import os

from qdrant_client import QdrantClient


if __name__=="__main__":
    print('Loading vector store...')

    vector_store_path = "./osha_app/src/qdrant_data"


    lock_file = os.path.join(vector_store_path, ".lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)

    os.makedirs(vector_store_path, exist_ok=True)

    qdrant_client = QdrantClient(path=vector_store_path)

    # TESTING WITH SMALL AMOUNT OF DATA
    print('Loading chunking data...')

    raw_data_path = './osha_app/src/ifixit data/Vehicle.json'


    with open(raw_data_path, "r", encoding="utf-8") as fh:
        chunker = ManualChunker(fh)
        dataset = chunker.chunk_data()

    print('Initializing predictor...')

    model = ExplainNLP.load_from_checkpoint('./osha_app/src/epoch=3-step=68672-001.ckpt')
    predictor = Predictor(
        model=model, 
        batch_size=3, 
        chunked_steps=dataset, 
        vector_client=qdrant_client
    )

    print('Predicting...')
    ret_dict = predictor.predict()
    print(ret_dict['predictions'])
    print(ret_dict['hypotheses'])
    print(ret_dict['premises'])
