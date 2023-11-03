import numpy as np

def document_vectorizer(tokens, wv, vector_size=300):
    '''
    Función que genera un vector para cada documento, sumando los vectores de cada palabra
    
    Parametros:
    -----------
    tokens: list
        Lista de tokens de cada documento
    wv: gensim.models.keyedvectors.KeyedVectors
        Modelo de word2vec entrenado
        
    Return:
    -------
    doc_vector: numpy.array
        Matriz con vectores de cada documento
    '''
    # Almacenamos los vectores de cada documento
    doc_vector = np.zeros(vector_size)
    
    # Num tokens en el doc
    n_words = 0
    
    # Sumamos los vectores de cada palabra
    for token in tokens:
        if token in wv:
            doc_vector += wv[token]
            n_words += 1

    # Promedio de los vectores dividido por el número de palabras
    if n_words > 0:
        doc_vector /= n_words

    return doc_vector