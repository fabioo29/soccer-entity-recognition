Preprocess text:
    Sub text >> UNK (numeros), NUM, PUNCT
    check stanza pipeline (processors='tokenize,mwt,pos,lemma')
    get short version of words and preprocess(label) text from those short words (stem)
        - avoid chesterfield != chesterfield's situations