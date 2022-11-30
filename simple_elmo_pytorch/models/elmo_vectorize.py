import numpy as np
import torch

from .elmo import batch_to_ids

class ElmoVectorizer(torch.nn.Module):
    
    def __init__(self, model, batch_size=32, device='cpu'):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.to(device)
        
        self.device = device
        self.batch_size = batch_size
        self.n_layers = model._elmo_lstm.num_layers
        self.vector_size = model.get_output_dim()
        
    def to(self, device):
        self.device = device
        self.model.to(device)
        
    def get_elmo_vectors(self, texts, warmup=True, layers="average"):
        """
        :param texts: list of sentences (lists of words)
        :param warmup: warm up the model before actual inference (by running it over the 1st batch)
        :param layers: ["top", "average", "all"].
        Yield the top ELMo layer, the average of all layers, or all layers as they are.
        :param session: external TensorFlow session to use
        :return: embedding tensor for all sentences
        (number of used layers by max word count by vector size)
        """
        max_text_length = max([len(t) for t in texts])
    
        # Creating the matrix which will eventually contain all embeddings from all batches:
        if layers == "all":
            final_vectors = np.zeros(
                (len(texts), self.n_layers, max_text_length, self.vector_size)
            )
        else:
            final_vectors = np.zeros((len(texts), max_text_length, self.vector_size))
            
        if warmup:
            self.warmup(texts)
    
        # Running batches:
        chunk_counter = 0
        for chunk in divide_chunks(texts, self.batch_size):
            # Converting sentences to character ids:
            
            sentence_ids = batch_to_ids(chunk)
            sentence_ids = sentence_ids.to(self.device)
            
            # Compute ELMo representations.
            with torch.no_grad():
                out = self.model(sentence_ids)
                elmo_vectors = out['elmo_representations'][0].detach().cpu().numpy()
            
            # Updating the full matrix:
            first_row = self.batch_size * chunk_counter
            last_row = first_row + elmo_vectors.shape[0]
            if layers == "all":
                final_vectors[
                first_row:last_row, :, : elmo_vectors.shape[2], :
                ] = elmo_vectors
            else:
                final_vectors[
                first_row:last_row, : elmo_vectors.shape[1], :
                ] = elmo_vectors
            chunk_counter += 1

        return final_vectors
    
    def get_elmo_vector_average(self, texts, warmup=True, layers="average"):
        """
        :param texts: list of sentences (lists of words)
        :param warmup: warm up the model before actual inference (by running it over the 1st batch)
        :param layers: ["top", "average", "all"].
        Yield the top ELMo layer, the average of all layers, or all layers as they are.
        :param session: external TensorFlow session to use
        :return: matrix of averaged embeddings for all sentences
        """
    
        if layers == "all":
            average_vectors = np.zeros((len(texts), self.n_layers, self.vector_size))
        else:
            average_vectors = np.zeros((len(texts), self.vector_size))
    
        counter = 0
    
        if warmup:
            self.warmup(texts)
    
        # Running batches:
        for chunk in divide_chunks(texts, self.batch_size):
            # Converting sentences to character ids:
            sentence_ids = batch_to_ids(chunk)
            sentence_ids = sentence_ids.to(self.device)
            
            with torch.no_grad():
                out = self.model(sentence_ids)
                elmo_vectors = out['elmo_representations'][0].detach().cpu().numpy()

            if layers == "all":
                elmo_vectors = elmo_vectors.reshape(
                    (
                        len(chunk),
                        elmo_vectors.shape[2],
                        self.n_layers,
                        self.vector_size,
                    )
                )
            for sentence in range(len(chunk)):
                if layers == "all":
                    sent_vec = np.zeros(
                        (elmo_vectors.shape[1], self.n_layers, self.vector_size)
                    )
                else:
                    sent_vec = np.zeros((elmo_vectors.shape[1], self.vector_size))
                for nr, word_vec in enumerate(elmo_vectors[sentence]):
                    sent_vec[nr] = word_vec
                semantic_fingerprint = np.sum(sent_vec, axis=0)
                semantic_fingerprint = np.divide(
                    semantic_fingerprint, sent_vec.shape[0]
                )
                query_vec = semantic_fingerprint / np.linalg.norm(
                    semantic_fingerprint
                )

                average_vectors[counter] = query_vec
                counter += 1
    
        return average_vectors


    def warmup(self, texts):
        for chunk0 in divide_chunks(texts, self.batch_size):
            #self.logger.info(f"Warming up ELMo on {len(chunk0)} sentences...")
            sentence_ids = batch_to_ids(chunk0)
            sentence_ids = sentence_ids.to(self.device)
            with torch.no_grad():
                _ = self.model(sentence_ids)
            #_ = sess.run(
                #self.elmo_sentence_input["weighted_op"],
                #feed_dict={self.sentence_character_ids: sentence_ids},
            #)
            break

def divide_chunks(data, n):
    for i in range(0, len(data), n):
        yield data[i: i + n]