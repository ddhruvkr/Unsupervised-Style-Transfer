model_config = {
    'dataset': '../../Data/v2/',
    'level': 'sentences',
    'MAX_LENGTH': 18,
    'embedding_dim': 300,
    'tag_dim': 100,
    'dep_dim': 100,
    'ver':'glove.6B.',
    'freq':0,
    'hidden_size': 256,
    'num_layers': 2,
    'dropout':0.2,
    'lr': 0.001,
    'classifer_class_size': 2,
    'model': 'Attn',
    'batch_size': 12,
    'epochs':100,
    'structural':False,
    'style': 'gender',#education, gender
    'classifier_name':'yelp_sentiment',
    'gpu':1,
    'dataset_source': 'PASTEL' #PASTEL, Yelp
}
