import torch

from simple_elmo_pytorch import ElmoBiLmSentence
from simple_elmo_pytorch import Elmo
from simple_elmo_pytorch import batch_to_ids

from simple_elmo_pytorch import ElmoVectorizer

import pickle

#model = ElmoBiLmSentence(options_file='/srv/dev-disk-by-uuid-fa7c0d11-e6eb-484b-858c-52bb87843d42/home/joefox/Downloads/elmo/196/options.json',
            #weight_file='/srv/dev-disk-by-uuid-fa7c0d11-e6eb-484b-858c-52bb87843d42/home/joefox/Downloads/elmo/196/model.hdf5',
            #)

model = Elmo(options_file='/srv/dev-disk-by-uuid-fa7c0d11-e6eb-484b-858c-52bb87843d42/home/joefox/Downloads/elmo/196/options.json',
            weight_file='/srv/dev-disk-by-uuid-fa7c0d11-e6eb-484b-858c-52bb87843d42/home/joefox/Downloads/elmo/196/model.hdf5',
            num_output_representations = 2
            )


sentence_ids = pickle.load(open('sentence_ids.pkl', 'rb'))
sentence_ids = torch.tensor(sentence_ids)

text_str = 'Привет моя строка как дела ура'

sentence_ids = batch_to_ids([text_str.split(),text_str.split(),text_str.split()])

model.eval()

out = model(sentence_ids)

vk = ElmoVectorizer(model, batch_size = 2)



print(model)
#print(model2)

model.eval()
model.get_output_dim()

text_str = 'Привет моя строка как дела ура'


v = vk.get_elmo_vector_average([text_str.split(' '), text_str.split(' '), text_str.split(' ')], warmup=True)

v = vk.get_elmo_vectors([text_str.split(' '), text_str.split(' '), text_str.split(' ')], warmup=True)



x = batch_to_ids([text_str.split()])


with torch.no_grad():
    res = model(x)
#model.reinit()
#res['elmo_representations'][0][0][0]
res['activations'][2][:,-1,:]


print('Ok!')