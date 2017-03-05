# coding=utf-8
import os, io, re, sys
import time
import chardet
import numpy as np
import array
import unicodedata
from random import shuffle
from gensim.models import Word2Vec
from helpers import *
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


""" Import the required packages. """
REQUIREMENTS = [ 'chardet', 'numpy', 'unicodedata', 'gensim', 'sklearn']
try:
    from setuptools import find_packages
    from distutils.core import setup
    from Cython.Distutils import build_ext as cython_build
    # import vobject
    from icalendar import Calendar, Event, vCalAddress, vText
except:
    import os, pip
    pip_args = [ '-vvv' ]
    proxy = os.environ['HTTP_PROXY']
    if proxy:
        pip_args.append('--proxy')
        pip_args.append(proxy)
    pip_args.append('install')
    for req in REQUIREMENTS:
        pip_args.append( req )
    print 'Installing requirements: ' + str(REQUIREMENTS)
    pip.main(initial_args = pip_args)

    # do it again
    import chardet
    import numpy as np
    import unicodedata
    from gensim.models import Word2Vec
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer


help_info = 'Usage:\npython word2vec_French [Model_Source] [What_To_Do]\n' \
            '1. For [Model Source], you can use \'Wiki\' for the model trained with Wikipedia, or \'Google\' for Google News.\n' \
            '2. For [What_To_Do], you can input \'evaluate\' to evaluate the performance, or \'demo\' to check the demo for extracting French addresses from French signatures\n' \



base_dir = os.path.abspath(os.path.dirname(__file__))
model_folder = os.path.join(base_dir, 'pretrained_model')
google_pretrained = os.path.join(model_folder, 'GoogleNews-vectors-negative300.bin')
wiki_pretrained = os.path.join(model_folder, 'Wiki_Giga.txt')
model_dict = {google_pretrained: 'normal_format_GoogleNews', wiki_pretrained: 'normal_format_Wiki'}
address_csv = os.path.join(base_dir, 'IQFrenchAddresses.csv')
enron_csv = os.path.join(base_dir, 'results_summary_cleaned&reviewed.csv')
labeled_French_email_csv = os.path.join(base_dir, 'Embedded_French_Addr_more.csv')
# labeled_French_email_csv = os.path.join(base_dir, 'Embedded_French_Addr.csv')
frenchSig100_csv = os.path.join(base_dir, 'FrenchAddress', '100_French_Signatures.csv')
batch_predicted_csv = os.path.join(base_dir, 'FrenchAddress', 'French_Address_Extracted.csv')
French_addresses_1000_csv = os.path.join(base_dir, '1000_French_Addresses.csv')

# vector
sentence_vectors_csv = os.path.join(base_dir, 'sentence_vectors', 'sv.csv')

delimiter = ' -$_$- '
vec2sentence_dict = {}
FN_dict = {}
FP_dict = {}
Missed_dict = {}
emails = getDictFromCsv(labeled_French_email_csv)['Email']
decoded = [x.decode(chardet.detect(x)['encoding']) for x in emails if x]
vectorizer = TfidfVectorizer(min_df=1)
tfidf = vectorizer.fit_transform(decoded).toarray()


def strip_accents(s):
    """
    remove the accent for French.
    :param s: French string with accents
    :return: French string without accents
    """
    s.replace('-', '_')
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def get_normal_format(model_file, binary_format=False, ):
    """
    transform the model that is directly downloaded from internet into a normal format that is able to load by simply calling the Word2Vec.load() function.
    """
    # loading from the directly download one.
    model = Word2Vec.load_word2vec_format(model_file, binary=binary_format)
    model.init_sims(replace=True)
    save_filename = os.path.join(model_folder, model_dict[model_file])
    model.save(save_filename)
    model = Word2Vec.load(save_filename, mmap='r')
    return model


def spliting(address):
    """
    split sentence into words and replace zipcode with france.
    :param address: input sentence
    :return: list of words
    """
    try:
        if re.search(r'\d{5}', address.decode('UTF-8-SIG')):
            address = re.sub(r'\d{5}', u'fran\xe7ais', address.decode('UTF-8-SIG'))
    except UnicodeDecodeError:
        encoding = chardet.detect(address)['encoding'] if chardet.detect(address)['encoding'] else 'ISO-8859-2'
        if re.search(r'\d{5}', address.decode(encoding)):
            address = re.sub(r'\d{5}', u'fran\xe7ais', address.decode(encoding))
    new = address.replace('-', ' ')
    return re.findall(r'[^\s!,.?":;0-9]+', new)


def getTfidfWeightedSentenceVector(sentence, doc_id):
    """
    tf-idf weighted version for function --->  getSentenceVector_2(sentence)
    :param sentence:
    :param doc_id:
    :return:
    """
    words = spliting(sentence)
    if not words:
        if len(''.join(re.findall('\d+', sentence)))/float(len(sentence)) > 0.8:
            words = u'numéro'
        else:
            # print 'No words found in sentence:\n' + sentence
            return None
    sentence_vec = np.zeros(shape=(300L,))
    added, missed = 0, 0
    encoding = chardet.detect(sentence)['encoding'] if chardet.detect(sentence)['encoding'] else 'utf8'
    for word in words:
        try:
            word = word.decode(encoding)
        except:
            word = unicode(word)
        try:
            vec = model[word]
            word_id = vectorizer.vocabulary_.get(unicode(word.lower()))
            weight = tfidf[doc_id][word_id] if word_id else 0
            check = vec == np.zeros(shape=(300L,))
            if not check.all() and weight == 0:
                weight = np.mean(tfidf)
            sentence_vec += weight * vec
            added += 1
        except KeyError:
            try:
                word_accents_removed = strip_accents(word)
                vec = model[word_accents_removed]
                weight = tfidf[doc_id][vectorizer.vocabulary_.get(unicode(word_accents_removed.lower()))]
                sentence_vec += weight * vec
                added += 1
            except KeyError:
                missed += 1
    try:
        p = missed / float(added + missed)
    except ZeroDivisionError:
        print 'aHa'
    if p > 0.8:
        Missed_dict[sentence] = Missed_dict[sentence] + 1 if Missed_dict.has_key(sentence) else 1
        # print 'Missed Percent: ' + str(p)
        # print sentence
        return None
    sentence_vec = sentence_vec.__div__(float(added))
    return sentence_vec


def getSentenceVector_2(sentence):
    """
    transform the sentence into a vector.
    :param sentence: one line of string
    :return: according vectors.
    """
    # preprosessing
    if re.search('[\w\.\+\-]+@[\w\.\+\-]+', sentence):
        return None
    if len(''.join(re.findall('\d+', sentence)))/float(len(sentence)) > 0.8 and len(''.join(re.findall('\d+', sentence))) != 5:
        return None
    sentence = re.sub('\d{5}', 'code postal', sentence)
    if not sentence:
        return None

    words = spliting(sentence)
    if not words and sentence:
        if len(''.join(re.findall('\d+', sentence)))/float(len(sentence)) > 0.8:
            words = u'numéro'
        else:
            # print 'No words found in sentence:\n' + sentence
            return None
    sentence_vec = np.zeros(shape=(300L,))
    added, missed = 0, 0
    encoding = chardet.detect(sentence)['encoding'] if chardet.detect(sentence)['encoding'] else 'utf8'
    for word in words:
        try:
            word = word.decode(encoding)
        except:
            word = unicode(word)
        try:
            sentence_vec += model[word]
            added += 1
        except KeyError:
            try:
                word_accents_removed = strip_accents(word)
                sentence_vec += model[word_accents_removed]
                added += 1
            except KeyError:
                missed += 1
    try:
        p = missed / float(added + missed)
    except ZeroDivisionError:
        print 'aHa'
    if p > 0.8:
        Missed_dict[sentence] = Missed_dict[sentence] + 1 if Missed_dict.has_key(sentence) else 1
        # print 'Missed Percent: ' + str(p)
        # print sentence
        return None
    sentence_vec = sentence_vec.__div__(float(added))
    return sentence_vec


def getVectors(model, sentences, tfidf_weight=False):
    """
    transform all the sentences into vectors using the model.
    :param model: pre-trained word2vec model
    :param sentences:
    :param tfidf_weight: default is False, which is to use average.
    :return: all the according vectors for the sentences.
    """
    vectors = []
    for sentence_docid in sentences:
        sentence = sentence_docid.split(delimiter)[0]
        doc_id = sentence_docid.split(delimiter)[1]
        if tfidf_weight:
            # do nothing at the moment
            sentence_vec = getTfidfWeightedSentenceVector(sentence, int(doc_id))
        else:
            sentence_vec = getSentenceVector_2(sentence)
        if sentence_vec is not None:
            vectors.append(sentence_vec)
            vec2sentence_dict.update({repr(sentence_vec): sentence})
    return vectors


def loadPretrainedModel(WVsource=google_pretrained, is_binary=True):
    """
    2 models basically, 1st is google which is binary, the other one is Wiki, which is not binary.
    :param WVsource: Google or Wiki?
    :param is_binary: True or False?
    :return: model
    """
    normal_format_model = os.path.join(model_folder, model_dict[WVsource])
    if not os.path.exists(normal_format_model):
        model = get_normal_format(WVsource, binary_format=is_binary)
    else:
        model = Word2Vec.load(normal_format_model, mmap='r')
    return model


def get_matrics(classifier, testing_vectors, testing_labels):
    """
    get the metrics for evaluating the the test data.
    :return: accuracy, precision, recall, f_score
    """
    predictions = []
    TP, TN, FP, FN = 0, 0 , 0, 0
    for index, vec in enumerate(testing_vectors):
        pre_label = classifier.predict(vec)
        predictions.append(pre_label)
        true_label = testing_labels[index]
        if true_label == 1 and pre_label == 1:
            TP += 1.0
        if true_label == 1 and pre_label == 0:
            FN += 1.0
            FN_dict[vec2sentence_dict[repr(vec)]] = FN_dict[vec2sentence_dict[repr(vec)]] + 1 if FN_dict.has_key(vec2sentence_dict[repr(vec)]) else 1
            # print "*** FN *** missed ture label: "
            # print vec2sentence_dict[repr(vec)] + '\n'
        if true_label == 0 and pre_label == 1:
            FP += 1.0
            FP_dict[vec2sentence_dict[repr(vec)]] = FP_dict[vec2sentence_dict[repr(vec)]] + 1 if FP_dict.has_key(vec2sentence_dict[repr(vec)]) else 1
            # print "*** FP *** wrong prediction: "
            # print vec2sentence_dict[repr(vec)] + '\n'
        if true_label == 0 and pre_label == 0:
            TN += 1.0
        try:
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            f_score = 2*precision*recall/(precision+recall)
        except ZeroDivisionError:
            continue
    return accuracy, precision, recall, f_score


def evaluate_performance(pos_vectors, neg_vectors, iteration, repetition, classifier):
    """
    This function is self-apparent.
    :return: Overall and detailed performance metric as a dict, which is also save into a separate csv file.
    """
    accuracy, precision, recall, f_score = [], [], [], []
    for i in xrange(iteration):
        shuffle(pos_vectors)
        shuffle(neg_vectors)
        for r in xrange(repetition):
            training_pos, testing_pos = split_ndarray(pos_vectors, repetition, r)
            training_neg, testing_neg = split_ndarray(neg_vectors, repetition, r)
            training_label_pos, training_label_neg = np.ones(shape=(len(training_pos))), np.zeros(shape=(len(training_neg)))
            testing_label_pos, testing_label_neg = np.ones(shape=(len(testing_pos))), np.zeros(shape=(len(testing_neg)))

            training_vectors = np.concatenate([training_pos, training_neg])
            testing_vectors = np.concatenate([testing_pos, testing_neg])
            training_labels = np.concatenate([training_label_pos, training_label_neg])
            testing_labels = np.concatenate([testing_label_pos, testing_label_neg])
            classifier.fit(X=training_vectors, y=training_labels)
            a, p, r, f = get_matrics(classifier, testing_vectors, testing_labels)
            accuracy.append(round(a, 3))
            precision.append(round(p, 3))
            recall.append(round(r, 3))
            f_score.append(round(f, 3))
    accuracy_median = np.median(accuracy)
    precision_median = np.median(precision)
    recall_median = np.median(recall)
    f_score_median = np.median(f_score)
    accuracy_iqr = np.subtract(*np.percentile(accuracy, [75, 25]))
    precision_iqr = np.subtract(*np.percentile(precision, [75, 25]))
    recall_iqr = np.subtract(*np.percentile(recall, [75, 25]))
    f_score_iqr = np.subtract(*np.percentile(f_score, [75, 25]))
    performance = {'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f_score': f_score,
                    'accuracy_median': round(accuracy_median, 3),
                    'precision_median': round(precision_median, 3),
                    'recall_median': round(recall_median, 3),
                    'f_score_median': round(f_score_median, 3),
                    'accuracy_iqr': round(accuracy_iqr, 3),
                    'precision_iqr': round(precision_iqr, 3),
                    'recall_iqr': round(recall_iqr, 3),
                    'f_score_iqr': round(f_score_iqr, 3)}
    return performance


def getPositiveNegiveLines(csv_file):
    """
    do what the name says.
    :param csv_file: cells under 'Labelled Email' are French Emails, where
            French addresses are labeled with '#ADDRESS#' at the begining.
    :return: positive and negative lines.
    """
    pos_lines, neg_lines = [], []
    pos_set, neg_set = set(), set()
    labelled_emails = getDictFromCsv(csv_file)['Labelled Email']
    for doc_id, email in enumerate(labelled_emails):
        for line in email.splitlines():
            line = line.strip()
            if not line:
                continue
            elif line.startswith('#ADDRESS#'):
                line = line[len('#ADDRESS#'):]
                if line not in pos_set:
                    pos_set.add(line)
                    pos_lines.append(line + delimiter + str(doc_id))
            elif line not in neg_set:
                neg_set.add(line)
                neg_lines.append(line + delimiter + str(doc_id))
    print 'positive lines: ' + str(len(pos_lines)) + '\t negative lines: ' + str(len(neg_lines))
    return pos_lines, neg_lines

def getMorePositiveLines(csv_file):
    """
    used to extract the 1,000 French addressed we got, which is saved into a seperate csv file.
    :param csv_file: each cell under 'French Address' is French address in multi-lines
    :return: All the lines under header 'French Address'.
    """
    pos_lines = []
    pos_set = set()
    french_addresses = getDictFromCsv(csv_file)['French Address']
    for doc_id, email in enumerate(french_addresses):
        for line in email.splitlines():
            line = line.strip()
            if not line:
                continue
            elif line not in pos_set:
                pos_set.add(line)
                pos_lines.append(line + delimiter + str(doc_id))
    print 'More positive lines: ' + str(len(pos_lines))
    return pos_lines


def vectorizeDocs():
    emails = getDictFromCsv(labeled_French_email_csv)['Email']
    decoded = [x.decode(chardet.detect(x)['encoding']) for x in emails]
    vectorizer = TfidfVectorizer(min_df=1)
    tfidf = vectorizer.fit_transform(decoded).toarray()
    print vectorizer.vocabulary_.get(u'salut')
    print 'haha'



def trainClassifier(classifier):
    """
    get all vectors and according labels ready. fit into the input classifier.
    :return: the trained classifier.
    """
    training_vectors = np.concatenate([pos_vectors, neg_vectors])
    training_label_pos, training_label_neg = np.ones(shape=(len(pos_vectors))), np.zeros(shape=(len(neg_vectors)))
    training_labels = np.concatenate([training_label_pos, training_label_neg])
    with open(sentence_vectors_csv, 'wb') as csvout:
        writer = csv.writer(csvout)
        for i, v in enumerate(training_vectors):
            l = v.tolist()
            l.append(int(training_labels[i]))
            writer.writerow(l)

    classifier.fit(X=training_vectors, y=training_labels)
    return classifier


def extractFrenchAddress(classifier, input_csv, output_csv):
    """
    for demo use
    :param classifier: the untrained one
    :param input_csv: a csv file with French Signatures for each column
    :param output_csv: each row with all the French addresses extracted from the according column.
    :return: the trained classifier.
    """
    with open(input_csv, 'rU') as csvInput:
        print 'Loading csv files.'
        reader = csv.DictReader(csvInput)
        fields = reader.fieldnames
        with open(output_csv, 'wb') as csvOut:
            fields.append('prediction')
            writer = csv.DictWriter(csvOut, fieldnames= fields)
            writer.writeheader()
            print 'Extracting French addresses..'
            for row in reader:
                sigs = row.get('Answer.Signature')
                prediction = []
                try:
                    for id, line in enumerate(sigs.splitlines()):
                        if not line.strip():
                            continue
                        else:
                            vector = getVectors(model, [line + delimiter + str(id)], tfidf_weight=False)
                            if not vector:
                                continue
                            elif classifier.predict(vector[0]) == 1:
                                prediction.append(line)
                except AttributeError:
                    print '?'
                if prediction:
                    french_address = '\n'.join(prediction)
                    row.update({'prediction': french_address})
                writer.writerow(row)
            print 'Done. File saved.'


def isFrAddr(sentences):
    """
    predict if the input String is a French address or not.
    """
    if not sentences:
        print 'Need a string as input.'
        return
    prediction = []
    for id, line in enumerate(sentences.splitlines()):
        if not line.strip():
            print 'Empty input. Need a string to predict'
            continue
        else:
            vector = getVectors(model, [line + delimiter + str(id)], tfidf_weight=False)
            if not vector:
                continue
            elif classifier.predict(vector[0]) == 1:
                prediction.append(line)
    if prediction:
        # print '*** French addresses founded: ***'
        # print '\n'.join(prediction)
        x = 1
    else:
        print 'Can\'t find French addresses.'


if __name__ == '__main__':
    paras = [['Wiki', 'Google'], ['evaluate', 'demo']]
    if len(sys.argv) != 3 or sys.argv[1] not in paras[0] or sys.argv[2] not in paras[1]:
        print help_info
        sys.exit()
    # load the pre-trained model
    start_time = time.time()
    if sys.argv[1] == 'Wiki':
        print 'the model is trained with Wikipedia'
        print 'loading models... Need some time here.'
        model = loadPretrainedModel(wiki_pretrained, is_binary=False)
    elif sys.argv[1] == 'Google':
        print 'the model is trained with Google News'
        print 'loading models... Need some time here.'
        model = loadPretrainedModel(google_pretrained, is_binary=True)
    print("--- %s seconds passed---" % (time.time() - start_time))
    print 'model loaded!'

    """
    Below is the previous version for only use French address as positive and English Email signature as negative.
    """
    #########################################################################################################################
    # # get the positive vectors, i.e. get the sentence_vectors for the French addresses
    # print 'transforming Franch Addresses to Positive Vectors...'
    # addresses = getDictFromCsv(address_csv).values()[0]
    # pos_vectors = getVectors(model, addresses)
    # print 'Successfully got Positive Vectors!'
    #
    # # get the negative vectors, i.e. get the sentence_vectors for the English Email Signatures at the moment...
    # print 'transforming English Email Signatures to Positive Vectors...'
    # English_Signatures = getDictFromCsv(enron_csv)['Full Signature']
    # sig_lines = [line for sig in English_Signatures if sig for line in sig.splitlines() if line]
    # neg_vectors = getVectors(model, sig_lines)
    # print 'Successfully got Negative Vectors!'
    # print 'Time spent: ' + str(time.time() - start_time)
    #########################################################################################################################


    # get the positive and negative vectors directly from the labeled French Emails.
    print 'Extracting positive and negative lines from labeled French Emails.'
    pos_lines, neg_lines = getPositiveNegiveLines(labeled_French_email_csv)
    print 'Adding the newly got 1,000 French Addresses into the positive lines.'
    more_pos_lines = getMorePositiveLines(French_addresses_1000_csv)
    pos_lines.extend(more_pos_lines)
    print 'transforming positive and negative sentence lines into vectors... Need some time here.'
    pos_vectors = getVectors(model, pos_lines, tfidf_weight=False)
    neg_vectors = getVectors(model, neg_lines, tfidf_weight=False)
    print 'Successfully got all Vectors!'
    print '--- Time spent: ' + str(time.time() - start_time) + ' seconds ---'


    if sys.argv[2] == 'demo':
        """ Below is used for IQ bi-week demo. """
        ###############################################################
        print 'demo begins. Results will be in the FrenchAddress folder. check it out there.'
        classifier = LinearSVC(C=10.0)
        classifier = trainClassifier(classifier)
        extractFrenchAddress(classifier, frenchSig100_csv, batch_predicted_csv)
        # isFrAddr('PÃ´le de Commerces et de Loisirs Confluence')
        ###############################################################
    elif sys.argv[2] == 'evaluate':
        # evaluate the performance
        print 'evaluating the performance...'
        iteration = 10
        repetition = 5
        classifier = LinearSVC(C=10.0)
        # the classifier hasn't been exposed to command line input yet.
        # classifier = RandomForestClassifier(n_estimators=10)
        start = time.time()
        performance_dict = evaluate_performance(pos_vectors, neg_vectors, iteration, repetition, classifier)
        performance_dict.update({'Missed French Addresses': FN_dict, 'Wrong Prediction Lines': FP_dict})
        if sys.argv[1] == 'Wiki':
            csv_name = 'preformance_' + 'Wiki_' + 'SVM_iter' + str(iteration) + '_rep' + str(repetition) + '.csv'
        elif sys.argv[1] == 'Google':
            csv_name = 'preformance_' + 'GoogleNews_' + 'SVM_iter' + str(iteration) + '_rep' + str(repetition) + '.csv'
        output_csv = os.path.join(base_dir, csv_name)
        saveDict2Csv(performance_dict, output_csv)
        print 'evaluation metrics save in: ' + csv_name
        print '--- Time spent on evaluation: ' + str(time.time() - start) + ' seconds ---'
